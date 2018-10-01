using Images, FileIO, Colors, FixedPointNumbers

# global variables. as a matter of convention, each one ends with an
# underscore. this is to make it harder to use them within the main body of the
# code without explicitly doing so.
# (width, height) of system in voxels
sys_size_ = (140,210);
# (dx, dy) voxel lengths
voxel_lengths_ = (0.12,0.12);
#time steps
dt_ = 1.;
#max time of simulation
max_t_ = 10000;
#min and max values of k and F
# #goes along y axis
# k_ = (0.0, 0.07);
# #goes along x axis
# F_ = (0.04, 0.16);
#goes along y axis
F_ = (0.0, 0.04);
#goes along x axis
k_ = (0.03, 0.07);
# #(Du Dv): diffusion constants 
# D_ = (2e-5, 1e-5);
#(Du Dv): diffusion constants 
D_ = (2e-3, 1e-3);
#proportion of squares to initialise
random_init_ = 0.99;
#choose boundary conditions, either dirichlet or neumann
diff_mode_ = "neumann";
checkpoint_int_ = 5;
checkpoint_name_ = "./gray_scott"

function create_initial_conditions((num_x, num_y), random_init)
    u = rand(num_x, num_y)
    @. u[u < random_init] = 0
    @. u[u >= random_init] = 1
    v = 1 .- u
    (u,v)
end;


function create_reaction_rates((min_k,max_k), (min_F, max_F), (num_x, num_y))
    ks = reshape(repeat(range(min_k, stop=max_k, length=num_x), num_y), (num_x,num_y))
    Fs = transpose(reshape(repeat(range(min_F, stop=max_F, length=num_y), num_x), (num_y,num_x)))
    reaction_rates = zeros(2, num_x, num_y)
    reaction_rates[1,:,:] = ks
    reaction_rates[2,:,:] = Fs
    reaction_rates
end;

function diffusion_voxel(i,j, dx_diff, a, (dx,dy), (num_x, num_y), dt, D, diff_mode)
    if diff_mode=="neumann"
        #diff left
        if i - 1 >= 1
            dx_diff[i-1,j] += a[i,j] * dt * D / (dx^2)
            dx_diff[i,j] -= a[i,j] * dt * D / (dx^2)
        end
        #diff right
        if i + 1 <= num_x
            dx_diff[i+1,j] += a[i,j] * dt * D / (dx^2)
            dx_diff[i,j] -= a[i,j] * dt * D / (dx^2)
        end
        #diff up
        if j - 1 >= 1
            dx_diff[i,j-1] += a[i,j] * dt * D / (dy^2)
            dx_diff[i,j] -= a[i,j] * dt * D / (dy^2)
        end
        #diff down
        if j + 1 <= num_y
            dx_diff[i,j+1] += a[i,j] * dt * D / (dy^2)
            dx_diff[i,j] -= a[i,j] * dt * D / (dy^2)
        end
    elseif diff_mode=="dirichlet"
        #diff left
        if i - 1 >= 1
            dx_diff[i-1,j] += a[i,j] * dt * D / (dx^2)
        end
        #diff right
        if i + 1 <= num_x
            dx_diff[i+1,j] += a[i,j] * dt * D / (dx^2)
        end
        #diff up
        if j - 1 >= 1
            dx_diff[i,j-1] += a[i,j] * dt * D / (dy^2)
        end
        #diff down
        if j + 1 <= num_y
            dx_diff[i,j+1] += a[i,j] * dt * D / (dy^2)
        end
        dx_diff[i,j] -= 4 * a[i,j] * dt * D / (dy^2)
    end
    dx_diff
end

function diffusion_species(a, D, (dx, dy), (num_x, num_y), dt, diff_mode)
    dx_diff = zeros(num_x, num_y)
    for i in 1:num_x
        for j in 1:num_y
            dx_diff = diffusion_voxel(i,j, dx_diff, a, (dx,dy), (num_x, num_y), dt, D, diff_mode)
        end
    end
    dx_diff
end

function diffusion_step((u,v), (Du, Dv), (dx, dy), (num_x, num_y), dt, diff_mode)
    du_diff = diffusion_species(u, Du, (dx, dy), (num_x, num_y), dt, diff_mode)
    dv_diff = diffusion_species(v, Dv, (dx, dy), (num_x, num_y), dt, diff_mode)
    (du_diff, dv_diff)
end

function reaction_step((u,v), reaction_rates, (dx,dy), (num_x, num_y), dt)
    ks = reaction_rates[1,:,:]
    Fs = reaction_rates[2,:,:]
    non_linear = zeros(num_x, num_y)
    du_reac = zeros(num_x, num_y)
    dv_reac = zeros(num_x, num_y)
    @. non_linear = (v^2) * u * dt
    @. du_reac = -non_linear + (Fs*(1-u) * dt)
    @. dv_reac = non_linear - ((ks + Fs) * v * dt)
    (du_reac, dv_reac)
end;

function pde_step!((u, v), D, reaction_rates, voxel_lengths, sys_size, dt, diff_mode)
    du_reac, dv_reac = reaction_step((u,v), reaction_rates, voxel_lengths, sys_size, dt)
    du_diff, dv_diff = diffusion_step((u,v), D, voxel_lengths, sys_size, dt, diff_mode)
    @. u += du_diff + du_reac
    @. v += dv_diff + dv_reac
    (u, v)
end;

function pixel_color_transform(pixel)
    output = zeros(3)
    #red to blue colormap
    # if pixel <= 0
        # output = RGB(1,0,0)
    # elseif pixel < 0.25
        # output = RGB(1-4pixel, 4pixel, 0)
    # elseif pixel <=0.5
        # output = RGB(0, 1-4(pixel-0.25), 4(pixel-0.25))
    # else
        # output = RGB(0,0,1)
    # end
    
    #white to blue colormap
    if pixel <= 0
        output = RGB(1.,1.,1.)
    elseif pixel <=0.5
        output = RGB(1-2pixel,1-2pixel,1)
    else
        output = RGB(0.,0.,1.)
    end
    output
end

function render_img((u,v))
    x,y = size(u)
    canvas = zeros(RGB,x,y)
    for i in 1:x
        for j in 1:y
            canvas[i,j] = pixel_color_transform(v[i,j])
        end
    end
    canvas
end

function set_up_simulate(max_t, dt, sys_size, random_init, k, F)
    num_steps = round(max_t/dt)
    u, v = create_initial_conditions(sys_size, random_init)
    reaction_rates = create_reaction_rates(k, F, sys_size)
    (num_steps, (u,v), reaction_rates)
end

function simulate!((u,v), D, reaction_rates, voxel_lengths, sys_size, dt, num_steps, diff_mode, checkpoint_int, checkpoint_name)
    x,y = size(u)
    num_frames = floor(Int, num_steps/checkpoint_int)
    index_count = 1
    for i in 1:num_steps
        u,v = pde_step!((u,v), D, reaction_rates, voxel_lengths, sys_size, dt, diff_mode)
        if ((i%checkpoint_int) - 1) == 0
            println("Processing step ", i, " out of ", num_steps)
            img = render_img((u,v))
            save(string(checkpoint_name,"_",index_count,".jpg"), img);
            index_count += 1
        end
    end
    (u,v)
end

function main(max_t, dt, sys_size, random_init, k, F, D, voxel_lengths, diff_mode, checkpoint_int, checkpoint_name)
    num_steps, (u,v), reaction_rates = set_up_simulate(max_t, dt, sys_size, random_init, k, F)
    u, v = simulate!((u,v), D, reaction_rates, voxel_lengths, sys_size, dt, num_steps, diff_mode, checkpoint_int, checkpoint_name)
end

main(max_t_, dt_, sys_size_, random_init_, k_, F_, D_, voxel_lengths_, diff_mode_, checkpoint_int_, checkpoint_name_)
