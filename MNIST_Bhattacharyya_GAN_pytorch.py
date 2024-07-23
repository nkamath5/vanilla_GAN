# MNIST image generation using GAN
from IPython.core.interactiveshell import dis
import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio


generator_gradient_dict = {}
discriminator_gradient_dict = {}
step_size = 1e-4
real_log_prob_estimator, min_sd_real, max_sd_real = None, None, None # will get inited in __main__


# De-normalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def get_log_prob(vectors):
    """ # inputs: vectors must be n x 1
        # returns: log probs estimated at a linspace of points using kde """

    scotts_factor = vectors.shape[0]**(-1.0/(4+vectors.shape[1])) # heuristic for bandwidth computation
    mean_vector   = vectors.mean(dim=0) # mean for each dimension/ col, so taken across rows
    normalized_vectors = vectors - mean_vector.unsqueeze(0)  # nxd - 1xd
    covariance    = torch.mm(torch.t(normalized_vectors), normalized_vectors) / (vectors.shape[0]-1) # (1xn) @ (nx1)
    bandwidth     = covariance*scotts_factor**2 *11 # *11 added because it just seems to work while comparing with scipy implementation (for aapm dataset)
    log_prob_estimator = kde.KernelDensityEstimator(vectors, kde.GaussianKernel(bandwidth.item()))
    return log_prob_estimator

def get_real_distribution(train_dataset, batch_size=64):
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    real_std_dev_list = []
    for real, _ in tqdm(data_loader):
      real = real.view(-1, 28*28).to(device)
      real = (real - real.min())/ (real.max() - real.min())
      real_std_dev_list.append(torch.std(real, dim=1))
    real_std_dev_tensor = torch.cat(real_std_dev_list)
    del real_std_dev_list
    import gc
    gc.collect()
    real_log_prob_estimator = get_log_prob(real_std_dev_tensor.view(-1,1))
    return real_log_prob_estimator, torch.min(real_std_dev_tensor).item(), torch.max(real_std_dev_tensor).item()

def bhattacharyya_distance(gen_vector, min_sd_real, max_sd_real):

    # print(f"min_sd_real, max_sd_real : {min_sd_real, max_sd_real}")
    min_sd_gen, max_sd_gen = torch.min(gen_vector).item(), torch.max(gen_vector).item()
    # print(f"gen batch min & max: {min_sd_gen, max_sd_gen}")

    combined_min, combined_max = min(min_sd_real, min_sd_gen), max(max_sd_real, max_sd_gen)

    N_STEPS = (combined_max - combined_min) // step_size
    xs = torch.linspace(combined_min, combined_max, int(N_STEPS), device=device)[:-1].reshape(-1,1)
    bin_width = step_size #1.0 * (max_cosine - min_cosine)/N_STEPS


    log_prob_dens_real = real_log_prob_estimator.forward(xs)
    gen_log_prob_estimator = get_log_prob(gen_vector.view(-1, 1))
    log_prob_dens_gen = gen_log_prob_estimator.forward(xs)

    # print(f"combined_min, combined_max: {combined_min, combined_max}")
    # print(f"first few log_prob_dens_real: {log_prob_dens_real[:10]}")
    # print(f"first few log_prob_dens_gen: {log_prob_dens_gen[:10]}")


    prob_dens_gen, prob_dens_real = torch.exp(log_prob_dens_gen), torch.exp(log_prob_dens_real)
    if not prob_dens_real.isfinite().all() or not prob_dens_gen.isfinite().all():
        print("prob densities of real or gen are not finite")
        breakpoint()
    return - 1.0 * torch.log((torch.sum(torch.sqrt((prob_dens_gen + 1e-6) * (prob_dens_real + 1e-6))) * step_size)) # 1e-6 to prevent disjoint support # revise eps to 1% of min prob seen otherwise

def pick_elements_and_gradients(tensor, num_elements=15):
    """
    Picks elements and their gradients from a tensor (parameter).
    Args:
        tensor: The input tensor.
        num_elements: The number of elements to pick.

    Returns:
        A tuple containing:
            - A tensor with the selected elements.
            - A tensor with the gradients of the selected elements.
    """

    # Don't select a subset of the weights; consider all
    if num_elements is None:
      return range(tensor.numel()), tensor.flatten(), tensor.grad.flatten()

    # Ensure the tensor has enough elements
    num_available = tensor.numel()
    num_to_pick = min(num_elements, num_available)

    # Get equally spaced indices based on numel
    indices = np.linspace(0, num_available - 1, num_to_pick, dtype=int)  # indices of the wts in the layer after flattening

    # Select elements and gradients
    selected_elements = tensor.flatten()[indices]
    selected_gradients = tensor.grad.flatten()[indices]
    return indices, selected_elements, selected_gradients

def check_gradients(generator, discriminator, D_input_dim, criterion, z_dim, batch_sizes, curr_kimg, device=torch.device('cpu')):
    print(f"Checking gradients across various batches of various batch sizes for curr_kimg: {curr_kimg}")

    # MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,), std=(0.5,))])
    
    mnist_data = dsets.MNIST(root=data_dir,
                             train=True,
                             transform=transform,
                             download=True)

    # save the GAN's weights
    torch.save({'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict()}, f'gan_checkpoint_{curr_kimg}_kimgs.pth')

    if curr_kimg not in generator_gradient_dict:
        generator_gradient_dict[curr_kimg] = {}
    if curr_kimg not in discriminator_gradient_dict:
        discriminator_gradient_dict[curr_kimg] = {}
    for batch_size in batch_sizes:

        print(f"Computing grads for batch size: {batch_size}")
        data_loader_for_grads = torch.utils.data.DataLoader(dataset=mnist_data,
                                              batch_size=batch_size,
                                              shuffle=True, drop_last=True)
        data_iter = iter(data_loader_for_grads)

        for itr in range(200):
            
            # generate fakes and run them thru D
            z = torch.randn(batch_size, z_dim, device=device)
            fakes = generator(z)
            D_fake_decision = discriminator(fakes)

            # zero out grads in G & D
            generator.zero_grad()
            discriminator.zero_grad()
            
            # loss.backward on G & save grads
            G_vanilla_loss = criterion(D_fake_decision, torch.ones_like(D_fake_decision))
            ## including Bhattacharyya Loss
            fake_std = torch.std(fakes.reshape(batch_size, -1), dim=1)
            Bhatt_loss = bhattacharyya_distance(fake_std, min_sd_real, max_sd_real)
            G_vanilla_loss.backward(retain_graph=True) # to prevent freeing up of intermediate tensors needed for D_fake_Loss.backward()
            Bhatt_loss.backward(retain_graph=True)

            picked = {}
            for name, param in generator.named_parameters():
                indices, selected_weights, selected_grads = pick_elements_and_gradients(param, 20)
                picked[name] = {
                    'indices': indices,
                    'weights': selected_weights,
                    f'grads_itr_{itr}': selected_grads
                }
            if itr == 0:
                generator_gradient_dict[curr_kimg][batch_size] = picked
            else:
                for layer_name in picked.keys():
                    generator_gradient_dict[curr_kimg][batch_size][layer_name][f'grads_itr_{itr}'] = picked[layer_name][f'grads_itr_{itr}']

            # zero out grads in D
            discriminator.zero_grad()

            # backward on losses of D & save grads
            D_fake_loss = criterion(D_fake_decision, torch.zeros_like(D_fake_decision))
            D_fake_loss.backward()

            try:
                real_imgs, _ = next(data_iter)
            except StopIteration:
              # if we run out of data, reinit the iterator
              data_iter = iter(data_loader_for_grads)
              real_imgs, _ = next(data_iter)
            D_real_decision = discriminator(real_imgs.view(-1, D_input_dim).to(device))
            D_real_loss = criterion(D_real_decision, torch.ones_like(D_fake_decision))
            D_real_loss.backward()

            picked = {}
            for name, param in discriminator.named_parameters():
              indices, selected_weights, selected_grads = pick_elements_and_gradients(param, 20)
              picked[name] = {
                'indices': indices,
                'weights': selected_weights,
                f'grads_itr_{itr}': selected_grads
              }
            if itr == 0:
                discriminator_gradient_dict[curr_kimg][batch_size] = picked
            else:
                for layer_name in picked.keys():
                    discriminator_gradient_dict[curr_kimg][batch_size][layer_name][f'grads_itr_{itr}'] = picked[layer_name][f'grads_itr_{itr}']
    
    torch.save(generator_gradient_dict, 'generator_gradient_dict.pth')
    torch.save(discriminator_gradient_dict, 'discriminator_gradient_dict.pth')


# Generator model
class Generator(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Generator, self).__init__()

        # Hidden layer
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(hidden_dims)):
            # Fully-connected layer
            fc_name = 'fc' + str(i+1)
            if i == 0:
                self.hidden_layer.add_module(fc_name, torch.nn.Linear(input_dim, hidden_dims[i], bias=True))
            else:
                self.hidden_layer.add_module(fc_name, torch.nn.Linear(hidden_dims[i-1], hidden_dims[i], bias=True))
            # Activation
            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

        # Output layer
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[i], output_dim, bias=True),
            torch.nn.Tanh()
        )

    def forward(self, x):
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out


# Discriminator model
class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Discriminator, self).__init__()

        # Hidden layer
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(hidden_dims)):
            # Fully-connected layer
            fc_name = 'fc' + str(i + 1)
            if i == 0:
                self.hidden_layer.add_module(fc_name, torch.nn.Linear(input_dim, hidden_dims[i], bias=True))
            else:
                self.hidden_layer.add_module(fc_name, torch.nn.Linear(hidden_dims[i-1], hidden_dims[i], bias=True))
            # Activation
            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))
            # Dropout
            drop_name = 'drop' + str(i + 1)
            self.hidden_layer.add_module(drop_name, torch.nn.Dropout(0.3))

        # Output layer
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[i], output_dim, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out


# Plot losses
def plot_loss(d_losses, g_losses, bh_losses, num_epoch, save=False, save_dir='MNIST_GAN_results/', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('Epoch {0}'.format(num_epoch + 1))
    plt.ylabel('Loss values')
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.plot(bh_losses, label='Bhattacharyya Loss')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'MNIST_GAN_losses_epoch_{:d}'.format(num_epoch + 1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def plot_result(generator, noise, num_epoch, save=False, save_dir='MNIST_GAN_results/', show=False, fig_size=(5, 5), device=torch.device('cpu')):
    generator.eval()

    noise = Variable(noise.to(device))
    gen_image = generator(noise)
    gen_image = denorm(gen_image)

    generator.train()

    n_rows = np.sqrt(noise.size()[0]).astype(np.int32)
    n_cols = np.sqrt(noise.size()[0]).astype(np.int32)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    for ax, img in zip(axes.flatten(), gen_image):
        ax.axis('off')
        ax.set_adjustable('box')
        ax.imshow(img.cpu().data.view(image_size, image_size).numpy(), cmap='gray', aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch+1)
    fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'MNIST_GAN_epoch_{:d}'.format(num_epoch+1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def setup_networks(G_input_dim, G_output_dim, D_input_dim, D_output_dim, hidden_dims, device=torch.device('cpu')):
    # Models
    G = Generator(G_input_dim, hidden_dims, G_output_dim).to(device)
    D = Discriminator(D_input_dim, hidden_dims[::-1], D_output_dim).to(device)
    return G, D

def setup_loss_optim():
    # Loss function
    criterion = torch.nn.BCELoss()
    
    # Optimizers
    G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)

    return criterion, G_optimizer, D_optimizer

def train_GAN(G, D, D_input_dim, criterion, G_optimizer, D_optimizer, data_loader, save_dir, device = torch.device('cpu'), check_grads=False, **check_grads_kwargs):
    kimg = 0.0

    # Training GAN
    D_avg_losses = []
    G_avg_losses = []
    Bh_avg_losses = []
    
    # Fixed noise for test
    num_test_samples = 5*5
    fixed_noise = torch.randn(num_test_samples, G_input_dim)
    
    for epoch in range(num_epochs):
        D_losses = []
        G_losses = []
        Bh_losses = []
    
        # minibatch training
        for i, (images, _) in enumerate(data_loader):
    
            # image data
            mini_batch = images.size()[0]
            x_ = images.view(-1, D_input_dim)
            x_ = Variable(x_.to(device))
    
            # labels
            y_real_ = Variable(torch.ones(mini_batch, 1).to(device))
            y_fake_ = Variable(torch.zeros(mini_batch, 1).to(device))
    
            # Train discriminator with real data
            D_real_decision = D(x_)
            # print(D_real_decision, y_real_)
            D_real_loss = criterion(D_real_decision, y_real_)
    
            # Train discriminator with fake data
            z_ = torch.randn(mini_batch, G_input_dim)
            z_ = Variable(z_.to(device))
            gen_image = G(z_)
    
            D_fake_decision = D(gen_image)
            D_fake_loss = criterion(D_fake_decision, y_fake_)
    
            # Back propagation
            D_loss = D_real_loss + D_fake_loss
            D.zero_grad()
            D_loss.backward()
            D_optimizer.step()
    
            # Train generator
            z_ = torch.randn(mini_batch, G_input_dim)
            z_ = Variable(z_.to(device))
            gen_image = G(z_)
    
            D_fake_decision = D(gen_image)
            G_vanilla_loss = criterion(D_fake_decision, y_real_)
            ## Bhattacharyya Loss
            fake_std = torch.std(gen_image.reshape(mini_batch, -1), dim=1)
            Bhatt_loss = bhattacharyya_distance(fake_std, min_sd_real, max_sd_real)
    
            # Back propagation
            D.zero_grad()
            G.zero_grad()
            G_vanilla_loss.backward()
            Bhatt_loss.backward()
            G_optimizer.step()
    
            # loss values
            D_losses.append(D_loss.item())
            G_losses.append(G_vanilla_loss.item())
            Bh_losses.append(Bhatt_loss.item())

            kimg += mini_batch / 1000.0
    
            print('Epoch [%d/%d], Step [%d/%d], kimg [%.2f], D_loss: %.4f, G_vanilla_loss: %.4f, Bhatt_loss: %.4f'
                  % (epoch+1, num_epochs, i+1, len(data_loader), kimg, D_loss.item(), G_vanilla_loss.item(), Bhatt_loss.item()))

            if check_grads and kimg > check_grads_kwargs['kimg_checkpoints'][0]:
                check_grads_kwargs['kimg_checkpoints'], check_grads = (check_grads_kwargs['kimg_checkpoints'][1:], True) if len(check_grads_kwargs['kimg_checkpoints'])>1 else ([], False)
                check_gradients(G, D, D_input_dim, criterion, G_input_dim, check_grads_kwargs['batch_sizes'], kimg, device)
    
        D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
        G_avg_loss = torch.mean(torch.FloatTensor(G_losses))
        Bh_avg_loss = torch.mean(torch.FloatTensor(Bh_losses))
    
        # avg loss values for plot
        D_avg_losses.append(D_avg_loss)
        G_avg_losses.append(G_avg_loss)
        Bh_avg_losses.append(Bh_avg_loss)
    
        plot_loss(D_avg_losses, G_avg_losses, Bh_avg_losses, epoch, save=True, save_dir=save_dir)
    
        # Show result for fixed noise
        plot_result(G, fixed_noise, epoch, save=True, fig_size=(5, 5), save_dir=save_dir, device=device)
    
    # Make gif
    loss_plots = []
    gen_image_plots = []
    for epoch in range(num_epochs):
        # plot for generating gif
        save_fn1 = save_dir + 'MNIST_GAN_losses_epoch_{:d}'.format(epoch + 1) + '.png'
        loss_plots.append(imageio.imread(save_fn1))
    
        save_fn2 = save_dir + 'MNIST_GAN_epoch_{:d}'.format(epoch + 1) + '.png'
        gen_image_plots.append(imageio.imread(save_fn2))
    
    imageio.mimsave(save_dir + 'MNIST_GAN_losses_epochs_{:d}'.format(num_epochs) + '.gif', loss_plots, fps=5)
    imageio.mimsave(save_dir + 'MNIST_GAN_epochs_{:d}'.format(num_epochs) + '.gif', gen_image_plots, fps=5)


if __name__ == "__main__":
    # Parameters
    image_size = 28
    G_input_dim = 100
    G_output_dim = image_size*image_size
    D_input_dim = image_size*image_size
    D_output_dim = 1
    hidden_dims = [256, 512, 1024]
    
    learning_rate = 0.0002
    batch_size = 128
    num_epochs = 100
    data_dir = '../Data/MNIST_data/'
    save_dir = 'MNIST_GAN_results/'
    
    # MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,), std=(0.5,))])
    
    mnist_data = dsets.MNIST(root=data_dir,
                             train=True,
                             transform=transform,
                             download=True)
    
    data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                              batch_size=batch_size,
                                              shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_log_prob_estimator, min_sd_real, max_sd_real = get_real_distribution(mnist_data)
    G, D = setup_networks(G_input_dim, G_output_dim, D_input_dim, D_output_dim, hidden_dims, device)
    criterion, G_optimizer, D_optimizer = setup_loss_optim()
    train_GAN(G, D, D_input_dim, criterion, G_optimizer, D_optimizer, data_loader, save_dir, device=device, check_grads=True, kimg_checkpoints=[0, 50, 500, 5000, 50000], batch_sizes=[32*2, 32*10, 32*50, 32*250])
    
