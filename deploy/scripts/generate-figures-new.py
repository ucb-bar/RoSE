from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from matplotlib import patches as patches
import pandas as pd
import matplotlib

import matplotlib.animation as animation

import numpy as np
font = {
        'size'   : 12}

plt.rc('font', **font)


#plt.rcParams['pdf.fonttype'] = 42
#plt.rcParams['ps.fonttype'] = 42
#plt.rcParams["text.usetex"] = True

def plot_trajectories_3d(ax, files, color, events=False):
    for file in files:
        traj = pd.read_csv(file)
        df = pd.DataFrame(traj, columns=['x', 'y', 'z', 'img_req'])

        # Data for a three-dimensional line
        x = -df.x.to_numpy()
        img_req = df.img_req.to_numpy()
        y = df.qy.to_numpy()
        z = -df.z.to_numpy()
        x = -df.x.to_numpy()
        ind = np.where(img_req > 0)

        ax.plot3D(x, y, z, color=color)

        x_req = np.take(x, ind)
        y_req = np.take(y, ind)
        z_req = np.take(z, ind)
        if events:
            ax.scatter3D(x_req, y_req, z_req, color=color)

def plot_trajectories_2d(ax, files, labels, colors, events=False, indices=None, path='straight', y_offset = 0, show_axes=True, show_legend=True, show_time=True):
    boundary1 = 1.7
    boundary2 = -1.5
    coords = [0, 50]
    if path =='straight':
        pass
        #ax.plot(coords,[boundary1, boundary1], "--",  color="grey", linewidth=1)
        #ax.plot(coords,[boundary2, boundary2], "--", color="grey", linewidth=1)
    else:
        a = np.arange(0, 2*np.pi, 0.01)
        b = np.sin(a)

        a = 80 / (2*np.pi) * a
        b = b * 7.5
        # ax.plot(a, -b+5.5, '--', color = 'grey', linewidth = 2, label = 'walls')
        # ax.plot(a, -b-8.5, '--', color = 'grey', linewidth = 2)
    for i, file in enumerate(files):
        traj = pd.read_csv(file)
        df = pd.DataFrame(traj, columns=['x', 'y', 'z', 'target_req', 'cycles', 'lin_x_vel', 'lin_y_vel', 'lin_z_vel'])
        cycles = df.cycles.to_numpy()
        # print(f"Cycles: {cycles}")
        cycle_diff = cycles[-1] - cycles[3]
        time_diff = cycle_diff / 1e9
        # Data for a three-dimensional line
        # x = -df.(x.to_numpy() + x_offset)
        img_req = df.target_req.to_numpy()
        y = df.y.to_numpy() - y_offset
        x = -df.x.to_numpy()

        x_vel = -df.lin_x_vel.to_numpy()
        y_vel =  df.lin_y_vel.to_numpy()
        vel = np.sqrt(x_vel ** 2 + y_vel ** 2)[2:]

        dist_vec = (np.sqrt((x[3:-1] - x[2:-2]) ** 2 + (y[3:-1] - y[2:-2]) ** 2))
        time_vec = dist_vec[2:] / vel[2:-2]
        time = np.sum(time_vec)
        dist = np.sum(dist_vec)
        # print(x_vel)
        avg_vel = np.mean(vel)
        
        ind = np.where(img_req > 0)
        if indices is None or i in indices:
            #ax.plot(x, y, color=colors[i], label=f"Config: {labels[i]}\nMission Time: {time_diff}s")
            #ax.plot(x, y, color=colors[i], label=f"{labels[i]}: {time_diff}s, {avg_vel:.2f}m/s, {dist:.2f}m, time: {time:.2f}")
            # ax.plot(x, y, color=colors[i], label=f"{labels[i]}:\n{time_diff}s, {avg_vel:.2f}m/s, {dist:.2f}m")
            if show_time:
                 ax.plot(x, y, color=colors[i], label=f"{labels[i]}:\n{time_diff}s")
            else:
                 ax.plot(x, y, color=colors[i], label=f"{labels[i]}")
            # ax.plot(x, y, color=colors[i], label=f"{labels[i]}s")
        
        x_req = np.take(x, ind)
        y_req = np.take(y, ind)
        if events and (indices is None or i in indices):
            print(ind)
            ax.scatter(x_req, y_req, color=colors[i])
        if show_axes:
            ax.set_xlabel("X Position (m)")
            ax.set_ylabel("Y Position (m)")

def plot_trajectories_2d_animated(fig, ax, files, labels, colors, events=False, indices=None, path='straight', y_offset = 0, show_axes=True, show_legend=True, show_time=True, legend_loc=None,legend_ncols=1, legend_bbox=None, title="", img=None, img_extent=None, grid=False):

    boundary1 = 1.7 
    boundary2 = -1.5
    coords = [0, 50]

    line, = ax.plot([], [], lw=2)

    #frames = 200
    frames = 100
    #frames = 5

    max_len = 0
    for file in files:
        traj = pd.read_csv(file)
        df = pd.DataFrame(traj, columns=['x', 'y', 'z', 'target_req', 'cycles', 'lin_x_vel', 'lin_y_vel', 'lin_z_vel'])
        if len(df) > max_len:
            max_len = len(df)
    step_size = int(max_len // (frames*0.8))

    # Temporarily plot all data to determine the limits
    for file in files:
        traj = pd.read_csv(file)
        df = pd.DataFrame(traj, columns=['x', 'y', 'z', 'target_req', 'cycles', 'lin_x_vel', 'lin_y_vel', 'lin_z_vel'])
        y = df.y.to_numpy() - y_offset
        x = -df.x.to_numpy()
        ax.plot(x, y)

    # Get the x and y limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()


    def animate(i):
        # Clear previous frame
        print(f"Processing frame {i}")
        ax.clear()

        # Iterate over files
        for j, file in enumerate(files):
            traj = pd.read_csv(file)
            df = pd.DataFrame(traj, columns=['x', 'y', 'z', 'target_req', 'cycles', 'lin_x_vel', 'lin_y_vel', 'lin_z_vel'])
            cycles = df.cycles.to_numpy()
            cycle_diff = cycles[-1] - cycles[3]
            time_diff = cycle_diff / 1e9
            img_req = df.target_req.to_numpy()
            y = df.y.to_numpy() - y_offset
            x = -df.x.to_numpy()

            # plot only the first i elements of the trajectory
            ax.plot(x[:min(i * step_size, len(x))], y[:min(i * step_size, len(x))], color=colors[j], label=f"{labels[j]}:\n{time_diff}s", linewidth=5)

            if show_axes:
                ax.set_xlabel("X Position (m)")
                ax.set_ylabel("Y Position (m)")
            else:
                plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
                for pos in ['right', 'top', 'bottom', 'left']:
                    plt.gca().spines[pos].set_visible(False)

        # Setting the x and y limits
        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)

        if img is not None:
            #ax.imshow(img, extent=img_extent)
            ax.imshow(img, extent=img_extent, aspect='auto')
        #ax.set_title(title)

        #ax.text(0.98, 0.86, title, transform=ax.transAxes, fontsize=12, ha='right', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        #ax.text(0.03, 0.87, title, transform=ax.transAxes, fontsize=12, ha='left', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        if grid:
            ax.grid()

        # Draw the legend
        ax.legend(loc=legend_loc, ncols=legend_ncols, bbox_to_anchor=legend_bbox)


    #plt.tight_layout()
    #plt.subplots_adjust(bottom=0.6, right=0.5, left=-0.1, top=1.0)
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=50)

    ani.save("trajectories.gif", writer='imagemagick', fps=20,dpi=150) 

    plt.show()

            
def plot_time_2d(ax, files, labels, colors, events=False, indices=None, path='straight', y_offset = 0):
    boundary1 = 1.6
    boundary2 = -1.6
    coords = [0, 50]
    if path =='straight':
        ax.plot(coords,[boundary1, boundary1], "--",  color="grey", linewidth=1)
        ax.plot(coords,[boundary2, boundary2], "--", color="grey", linewidth=1)
    else:
        a = np.arange(0, 2*np.pi, 0.01)
        b = np.sin(a)

        a = 80 / (2*np.pi) * a
        b = b * 7.5
        # ax.plot(a, -b+5.5, '--', color = 'grey', linewidth = 2, label = 'walls')
        # ax.plot(a, -b-8.5, '--', color = 'grey', linewidth = 2)
    for i, file in enumerate(files):
        traj = pd.read_csv(file)
        df = pd.DataFrame(traj, columns=['x', 'y', 'z', 'target_req', 'cycles', 'lin_x_vel', 'lin_y_vel', 'lin_z_vel'])
        cycles = df.cycles.to_numpy()
        times = cycles / 1e9
        # print(f"Cycles: {cycles}")
        cycle_diff = cycles[-1] - cycles[3]
        time_diff = cycle_diff / 1e9
        # Data for a three-dimensional line
        # x = -df.(x.to_numpy() + x_offset)
        img_req = df.target_req.to_numpy()
        y = df.y.to_numpy() - y_offset
        x = -df.x.to_numpy()

        x_vel = -df.lin_x_vel.to_numpy()
        y_vel =  df.lin_y_vel.to_numpy()
        vel = np.sqrt(x_vel ** 2 + y_vel ** 2)[2:]

        dist_vec = (np.sqrt((x[3:-1] - x[2:-2]) ** 2 + (y[3:-1] - y[2:-2]) ** 2))
        time_vec = dist_vec[2:] / vel[2:-2]
        time = np.sum(time_vec)
        dist = np.sum(dist_vec)
        # print(x_vel)
        avg_vel = np.mean(vel)
        
        ind = np.where(img_req > 0)
        if indices is None or i in indices:
            #ax.plot(x, y, color=colors[i], label=f"Config: {labels[i]}\nMission Time: {time_diff}s")
            #ax.plot(x, y, color=colors[i], label=f"{labels[i]}: {time_diff}s, {avg_vel:.2f}m/s, {dist:.2f}m, time: {time:.2f}")
            # ax.plot(x, y, color=colors[i], label=f"{labels[i]}:\n{time_diff}s, {avg_vel:.2f}m/s, {dist:.2f}m")
            #ax.plot(times - times[4], x, color=colors[i], label=f"{labels[i]}: {time_diff}s")
            ax.plot(times - times[4], x, color=colors[i], label=f"{labels[i]}")
            ax.set
            # ax.plot(x, y, color=colors[i], label=f"{labels[i]}s")
        
        x_req = np.take(x, ind)
        y_req = np.take(y, ind)
        if events and (indices is None or i in indices):
            ax.scatter(x_req, y_req, color=colors[i])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("X Position (m)")

def plot_deadlines_2d(ax, files, labels, colors, events=False, indices=None):
    for i, file in enumerate(files):
        traj = pd.read_csv(file)
        plt.ylim(0, 15)
        df = pd.DataFrame(traj, columns=['x', 'y', 'z', 'lin_x_vel', 'lin_y_vel', 'lin_z_vel', 'depth', 'target_req', 'cycles'])
        cycles = df.cycles.to_numpy()
        cycle_diff = cycles[-1] - cycles[2]
        time_diff = cycle_diff / 1e9
        # Data for a three-dimensional line
        x = -df.x.to_numpy()[10:]
        x_vel = -df.lin_x_vel.to_numpy()
        y_vel = df.lin_y_vel.to_numpy()
        z_vel = -df.lin_z_vel.to_numpy()
        vel = np.maximum(np.sqrt(x_vel**2 + y_vel**2 + z_vel**2), 0.1)
        depth = df.depth.to_numpy()
        deadline = depth/vel
        deadline = deadline - 0.007 - 0.1
        deadline = deadline[10:]
        if indices is None or i in indices:
            ax.plot(x, deadline, color=colors[i],label=f"Config: {labels[i]}\nMission Time: {time_diff}s")

def plot_deadline_violations_2d(ax, files, labels, colors, latencies, events=False, indices=None):
    for i, file in enumerate(files):
        traj = pd.read_csv(file)
        df = pd.DataFrame(traj, columns=['x', 'y', 'z', 'lin_x_vel', 'lin_y_vel', 'lin_z_vel', 'depth', 'target_req', 'cycles'])
        cycles = df.cycles.to_numpy()
        cycle_diff = cycles[-1] - cycles[2]
        time_diff = cycle_diff / 1e9
        # Data for a three-dimensional line
        x = -df.x.to_numpy()
        x_vel = -df.lin_x_vel.to_numpy()
        y_vel = df.lin_y_vel.to_numpy()
        z_vel = -df.lin_z_vel.to_numpy()
        vel = np.sqrt(x_vel**2 + y_vel**2 + z_vel**2)
        depth = df.depth.to_numpy()
        deadline = depth/vel
        violation = deadline < latencies[i] 
        if indices is None or i in indices:
            ax.plot(x,violation,  color=colors[i],label=f"Config: {labels[i]}\nMission Time: {time_diff}s")

def mse(base, approx, scale):
    ap_interp = np.array([])
    for i in range(3):
        approx[i]=approx[i][2:].copy()
        base[i]=base[i][2:].copy()
        print(f"base[{i}]: {base[i]}")
        print(f"approx[{i}]: {approx[i]}")
        approx[i] = approx[i][:len(base[i]) * scale]
        interp = np.interp(np.arange(len(base[i])), np.arange(len(approx[i])) * scale , approx[i])
        ap_interp = np.concatenate((ap_interp,interp))
    base_flat = np.concatenate((base[0], base[1], base[2]))
    print(f"base_flat: {base_flat}")
    print(f"ap_interp: {ap_interp}")
    mse = ((base_flat - ap_interp) ** 2).sum() / len(base)
    return mse 

def calculate_mses(files, scales):
    traj0 = pd.read_csv(files[0])
    df0 = pd.DataFrame(traj0, columns=['x', 'y', 'z'])

    # Data for a three-dimensional line
    x0 = -df0.x.to_numpy()
    y0 = df0.y.to_numpy()
    z0 = -df0.z.to_numpy()
    mses = []
    for i, file in enumerate(files):
        traj = pd.read_csv(file)
        df = pd.DataFrame(traj, columns=['x', 'y', 'z'])

        # Data for a three-dimensional line
        x = -df.x.to_numpy()
        y = df.y.to_numpy()
        z = -df.z.to_numpy()
        mses.append(mse([x0, y0, z0], [x,y,z], scales[i]))
    return mses

def plot_trajectories_dynamic(ax, files, labels, colors, events=False, indices=None, path='straight'):
    boundary1 = 1.6
    boundary2 = -1.6
    coords = [0, 50]
    if path =='straight':
        ax.plot(coords,[boundary1, boundary1], "--",  color="grey", linewidth=1)
        ax.plot(coords,[boundary2, boundary2], "--", color="grey", linewidth=1)
    else:
        a = np.arange(0, 2*np.pi, 0.01)
        b = np.sin(a)

        a = 80 / (2*np.pi) * a
        b = b * 7.5
        # ax.plot(a, -b+5.5, '--', color = 'grey', linewidth = 2, label = 'walls')
        # ax.plot(a, -b-8.5, '--', color = 'grey', linewidth = 2)
    for i, file in enumerate(files):
        traj = pd.read_csv(file)
        df = pd.DataFrame(traj, columns=['x', 'y', 'z', 'target_req', 'cycles', 'lin_x_vel', 'lin_y_vel', 'lin_z_vel', 'depth'])
        cycles = df.cycles.to_numpy()
        depth = df.depth.to_numpy()
        # print(f"Cycles: {cycles}")
        cycle_diff = cycles[-1] - cycles[3]
        time_diff = cycle_diff / 1e9
        # Data for a three-dimensional line
        x = -df.x.to_numpy()
        img_req = df.target_req.to_numpy()
        y = df.y.to_numpy()
        x = -df.x.to_numpy()

        x_vel = -df.lin_x_vel.to_numpy()
        y_vel =  df.lin_y_vel.to_numpy()
        vel = np.sqrt(x_vel ** 2 + y_vel ** 2)[2:]

        dist_vec = (np.sqrt((x[3:-1] - x[2:-2]) ** 2 + (y[3:-1] - y[2:-2]) ** 2))
        time_vec = dist_vec[2:] / vel[2:-2]
        time = np.sum(time_vec)
        dist = np.sum(dist_vec)
        # print(x_vel)
        avg_vel = np.mean(vel)
        
        ind = np.where(img_req > 0)
        if indices is None or i in indices:
            #ax.plot(x, y, color=colors[i], label=f"Config: {labels[i]}\nMission Time: {time_diff}s")
            #ax.plot(x, y, color=colors[i], label=f"{labels[i]}: {time_diff}s, {avg_vel:.2f}m/s, {dist:.2f}m, time: {time:.2f}")
            ax.plot(x, y, color=colors[i], label=f"{labels[i]}:\n{time_diff}s, {avg_vel:.2f}m/s, {dist:.2f}m")
            if i == 2:
                ax.plot(x, depth < 10.0, color=colors[i], label=f"{labels[i]}:\n{time_diff}s, {avg_vel:.2f}m/s, {dist:.2f}m")
            # ax.plot(x, y, color=colors[i], label=f"{labels[i]}s")
        
        x_req = np.take(x, ind)
        y_req = np.take(y, ind)
        if events and (indices is None or i in indices):
            ax.scatter(x_req, y_req, color=colors[i])
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")

def plot_analysis_dynamic(ax, files, labels, colors, latencies=[], events=False, indices=None, path='straight'):
    boundary1 = 1.6
    boundary2 = -1.6
    coords = [0, 50]
    for i, file in enumerate(files):
        traj = pd.read_csv(file)
        df = pd.DataFrame(traj, columns=['x', 'y', 'z', 'target_req', 'cycles', 'lin_x_vel', 'lin_y_vel', 'lin_z_vel', 'depth'])
        cycles = df.cycles.to_numpy()
        depth = df.depth.to_numpy()
        cycle_diff = cycles[-1] - cycles[3]
        time_diff = cycle_diff / 1e9
        x = -df.x.to_numpy()
        img_req = df.target_req.to_numpy()
        img_req[np.isnan(img_req)] = 0
        runtimes = img_req * (depth < 10) * latencies[i][0] + img_req * (depth >= 10) * latencies[i][1]
        # print(img_req)
        y = df.y.to_numpy()
        x = -df.x.to_numpy()



        x_vel = -df.lin_x_vel.to_numpy()
        y_vel =  df.lin_y_vel.to_numpy()
        vel = np.sqrt(x_vel ** 2 + y_vel ** 2)[2:]

        dist_vec = (np.sqrt((x[3:-1] - x[2:-2]) ** 2 + (y[3:-1] - y[2:-2]) ** 2))
        time_vec = dist_vec[2:] / vel[2:-2]
        time = np.sum(time_vec)
        dist = np.sum(dist_vec)
        # print(x_vel)
        avg_vel = np.mean(vel)
        
        ind = np.where(img_req > 0)
        if indices is None or i in indices:
            #ax.plot(x, y, color=colors[i], label=f"Config: {labels[i]}\nMission Time: {time_diff}s")
            #ax.plot(x, y, color=colors[i], label=f"{labels[i]}: {time_diff}s, {avg_vel:.2f}m/s, {dist:.2f}m, time: {time:.2f}")
            # ax.plot(x, y, color=colors[i], label=f"{labels[i]}:\n{time_diff}s, {avg_vel:.2f}m/s, {dist:.2f}m")
            # ax.plot(x, depth < 10.0, color=colors[i], label=f"{labels[i]}:\n{time_diff}s, {avg_vel:.2f}m/s, {dist:.2f}m")
            print(f"Total inferences for {labels[i]}: {np.sum(img_req)}")
            print(f"Total runtimes for {labels[i]}: {np.sum(runtimes)}")
            
            utilization = np.sum(runtimes) / time_diff
            print(f"Total utilizaition for {labels[i]}: {utilization}")
            if("ResNet14" in labels[i]):
                ax.axhline(y=utilization, color='k', linestyle='-', linewidth=1)
                ax.axvline(x=time_diff, color='k', linestyle='-', linewidth=1)
            if("Dynamic" in labels[i]):
                ax.scatter(time_diff, utilization, label=f"{labels[i]}", marker=(5,1), s=150)
            else:
                ax.scatter(time_diff, utilization, label=f"{labels[i]}", s=150)
            # ax.plot(x, y, color=colors[i], label=f"{labels[i]}s")
        
        x_req = np.take(x, ind)
        y_req = np.take(y, ind)
        if events and (indices is None or i in indices):
            ax.scatter(x_req, y_req, color=colors[i])
        ax.set_ylabel("Accelerator Activity Factor")
        ax.set_xlabel("Application Latency (s)")
        ax.set_ylim([0.6,1])
        ax.set_xlim([11,17])

def plot_rollouts_roofline(files, indices):
    perf_data = pd.read_csv (r'./deploy/hephaestus/logs/rose-perf-sync-only.csv')
    # print (perf_data)
    f = plt.figure()
    f.set_figwidth(7)
    f.set_figheight(4)
    x = np.linspace(1e4, 5e8,10)
    syncbound = 100*x/25
    fsim_throughput = np.ones(len(x))*4.5e7
    plt.loglog(x,syncbound, 'r--', label="Synchronization Bottleneck")
    plt.loglog(x,fsim_throughput, 'g--', label="FPGA Throughput")
    plt.loglog(perf_data[' FireSim Step'][:-3], perf_data[' Throughput (cycles/sec)'][:-3], 'b', label="Co-Simulation Throughput")
    
    throughputs = []
    for i, file in enumerate(files):
        traj = pd.read_csv(file)
        df = pd.DataFrame(traj, columns=['cycles', 'real_time'])
        # Data for a three-dimensional line
        cycles = df.cycles.to_numpy()[-1]-df.cycles.to_numpy()[3]
        real_time = df.real_time.to_numpy()[-1] - df.real_time.to_numpy()[3]
        print(f"cycles: {cycles}, real_time: {real_time}")
        throughput = cycles/real_time
        throughputs.append(throughput)
        
    # plt.loglog(x,syncbound, 'r--', label="Synchronization Bottleneck")
    # plt.loglog(x,fsim_throughput, 'g--', label="FPGA Throughput")
    #plt.ylim(1e4, 5e10)
    plt.loglog(indices, throughputs, 's', label="Co-Simulation Throughput With Application")
    plt.title("RoSÉ Sim. Throughput vs Synchronization Granularity", fontsize=14)
    plt.xlabel("SoC Cycles per Synchronization")
    plt.ylabel("RTL Sim. Throughput, FPGA (Hz)")
    plt.grid()
    plt.legend(loc='lower center', ncols=1, bbox_to_anchor=(0.4,0.625), fontsize=12)
    plt.savefig("./deploy/figures/figure15.pdf", bbox_inches='tight')
    plt.show()

def plot_hw_sw_sweep_bars(files, networks, cores, latencies=[]):
    boundary1 = 1.6
    boundary2 = -1.6
    coords = [0, 50]
    data_time = {'Network': networks}
    data_vel =  {'Network': networks}
    data_util = {'Network': networks}
    #for c in range(len(cores)):
    #for c in [1,0]:
    for c in [1,0]:
        data_time[cores[c]] = []
        data_vel[cores[c]]  = []
        data_util[cores[c]] = []
        for n in range(len(networks)):
            traj = pd.read_csv(files[n][c])
            df = pd.DataFrame(traj, columns=['x', 'y', 'z', 'target_req', 'cycles', 'lin_x_vel', 'lin_y_vel', 'lin_z_vel', 'depth'])
            cycles = df.cycles.to_numpy()
            depth = df.depth.to_numpy()
            cycle_diff = cycles[-1] - cycles[3]
            time_diff = cycle_diff / 1e9
            x = -df.x.to_numpy()
            img_req = df.target_req.to_numpy()
            img_req[np.isnan(img_req)] = 0
            runtimes = img_req * (depth < 10) * latencies[c][n][0] + img_req * (depth >= 10) * latencies[c][n][1]

            y = df.y.to_numpy()
            x = -df.x.to_numpy()

            x_vel = -df.lin_x_vel.to_numpy()
            y_vel =  df.lin_y_vel.to_numpy()
            vel = np.sqrt(x_vel ** 2 + y_vel ** 2)[2:]

            dist_vec = (np.sqrt((x[3:-1] - x[2:-2]) ** 2 + (y[3:-1] - y[2:-2]) ** 2))
            time_vec = dist_vec[2:] / vel[2:-2]
            time = np.sum(time_vec)
            data_time[cores[c]].append(time_diff)
            dist = np.sum(dist_vec)
            # print(x_vel)
            #avg_vel = np.mean(vel)
            avg_vel = dist / time
            data_vel[cores[c]].append(avg_vel)
            
            ind = np.where(img_req > 0)
            #utilization = np.sum(runtimes) / time_diff
            utilization = 100*np.sum(runtimes) / time_diff
            if c == 1:
                utilization = 0
            data_util[cores[c]].append(utilization)

    #fig, axs = plt.subplots(1, 3, figsize=(8, 3))
    fig, axs = plt.subplots(1, 2, figsize=(7, 2))

    df1 = pd.DataFrame(data_time)
    df1.plot(kind='bar', x='Network', ax=axs[0], fontsize=12)
    axs[0].legend('', frameon=False)
    axs[0].set_title('Mission Time (s)', fontsize=12)
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45) # rotate tick labels for the first subplot
    axs[0].set_xlabel('')
    axs[0].set_ylim([0, 40]) # set y-limits for the first subplot
    # df2 = pd.DataFrame(data_vel)
    # df2.plot(kind='bar', x='Network', ax=axs[1], fontsize=12)
    # axs[1].legend('', frameon=False)
    # axs[1].set_title('Avg Velocity (m/s)', fontsize=12)
    # axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45) # rotate tick labels for the first subplot
    # axs[1].set_xlabel('')
    # axs[1].set_ylim([0, 10]) # set y-limits for the first subplot
    # df3 = pd.DataFrame(data_util)
    # df3.plot(kind='bar', x='Network',  ax=axs[2], fontsize=12)
    # axs[2].set_title('Accelerator Activity Factor', fontsize=12)
    # #axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    # axs[2].legend(loc='lower center', ncols=2, bbox_to_anchor=(-0.7, -0.6))
    # axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=45) # rotate tick labels for the first subplot
    # axs[2].set_xlabel('')
    # axs[2].set_ylim([0, 1]) # set y-limits for the first subplot

    df2 = pd.DataFrame(data_util)
    df2.plot(kind='bar', x='Network',  ax=axs[1], fontsize=12)
    axs[1].set_title('Activity Factor (%)', fontsize=12)
    #axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    axs[1].legend(loc='lower center', ncols=2, bbox_to_anchor=(-0.10, -0.9))
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45) # rotate tick labels for the first subplot
    axs[1].set_xlabel('')
    #axs[1].set_ylim([0.75, 1]) # set y-limits for the first subplot
    axs[1].set_ylim([75, 100]) # set y-limits for the first subplot

    plt.savefig("./deploy/figures/figure14.pdf",  bbox_inches='tight')
    

# def plot_figure_10():
#     colors = ['#AB2328', '#4169E1', '#046A38']

#     gemmini_boom_files = [
#         r'./deploy/hephaestus/logs/tunnel-exp-boom-gemmini-160.csv', 
#         r'./deploy/hephaestus/logs/tunnel-exp-boom-gemmini-180.csv',
#         r'./deploy/hephaestus/logs/tunnel-exp-boom-gemmini-200.csv'
#     ]
#     gemmini_boom_labels = ['BOOM + Gemmini, -20°', 'BOOM + Gemmini, 0°', 'BOOM + Gemmini, 20°']

#     boom_files = [
#         r'./deploy/hephaestus/logs/tunnel-exp-boom-only-160.csv', 
#         r'./deploy/hephaestus/logs/tunnel-exp-boom-only-180.csv',
#         r'./deploy/hephaestus/logs/tunnel-exp-boom-only-200.csv'
#     ]
#     boom_labels = ['BOOM, -20°', 'BOOM, 0°', 'BOOM, 20°']

#     gemmini_rocket_files = [
#         r'./deploy/hephaestus/logs/tunnel-exp-rocket-gemmini-160.csv', 
#         r'./deploy/hephaestus/logs/tunnel-exp-rocket-gemmini-180.csv',
#         r'./deploy/hephaestus/logs/tunnel-exp-rocket-gemmini-200.csv'
#     ]
#     gemmini_rocket_labels = ['Rocket + Gemmini, -20°','Rocket + Gemmini, 0°', 'Rocket + Gemmini, 20°']
  
#     for d in [(gemmini_boom_files, gemmini_boom_labels, "a"), (gemmini_rocket_files, gemmini_rocket_labels, "b"), (boom_files, boom_labels, "c")]:      
#         f = plt.figure()
#         f.set_figwidth(5)
#         f.set_figheight(3)
#         # ax = plt.axes(projection='3d')
#         boundary1 = 1.6
#         boundary2 = -1.6
#         ax = plt.axes()
#         # plot_trajectories_2d(ax, gemmini_boom_files, gemmini_boom_labels, gemmini_boom_colors, events=False)
#         # plot_trajectories_2d(ax, boom_files, boom_labels, boom_colors, events=False)
#         plot_trajectories_2d(ax, d[0], d[1], colors, events=False, y_offset=-22.85)
#         ax.set_ylim(-2, 2)
#         plt.xlabel("Distance (m)")
#         # plt.ylabel("Lateral Offset (m)")
#         # plt.title("Rollouts for Different HW Configurations")
#         plt.grid()
#         plt.legend(loc='upper left', bbox_to_anchor=(1, 0.70))
#         # ax.set_zlabel("Altitude (m)")
#         plt.savefig(f"./deploy/figures/figure10_{d[2]}.png", bbox_inches = 'tight')
#         plt.show()

def plot_figure_10():
    colors = ['#AB2328', '#4169E1', '#046A38']
    titles = ['BOOM + Gemmini', 'Rocket + Gemmini', 'BOOM']
    legends = ['-20°', '0°', '20°']


    gemmini_boom_files = [
        r'./deploy/hephaestus/logs/tunnel-exp-boom-gemmini-160.csv', 
        r'./deploy/hephaestus/logs/tunnel-exp-boom-gemmini-180.csv',
        r'./deploy/hephaestus/logs/tunnel-exp-boom-gemmini-200.csv'
    ]
    gemmini_boom_labels = ['BOOM + Gemmini, -20°', 'BOOM + Gemmini, 0°', 'BOOM + Gemmini, 20°']

    boom_files = [
        r'./deploy/hephaestus/logs/tunnel-exp-boom-only-160.csv', 
        r'./deploy/hephaestus/logs/tunnel-exp-boom-only-180.csv',
        r'./deploy/hephaestus/logs/tunnel-exp-boom-only-200.csv'
    ]
    boom_labels = ['BOOM, -20°', 'BOOM, 0°', 'BOOM, 20°']

    gemmini_rocket_files = [
        r'./deploy/hephaestus/logs/tunnel-exp-rocket-gemmini-160.csv', 
        r'./deploy/hephaestus/logs/tunnel-exp-rocket-gemmini-180.csv',
        r'./deploy/hephaestus/logs/tunnel-exp-rocket-gemmini-200.csv'
    ]
    gemmini_rocket_labels = ['Rocket + Gemmini, -20°','Rocket + Gemmini, 0°', 'Rocket + Gemmini, 20°']
  
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(5, 8))

    for i, d in enumerate([(gemmini_boom_files, gemmini_boom_labels, colors), (gemmini_rocket_files, gemmini_rocket_labels, colors), (boom_files, boom_labels, colors)]):
        ax_i = axes[i]
        plot_trajectories_2d(ax_i, d[0], d[1], d[2], events=False, y_offset=-22.85, show_axes=False)
        img = mpimg.imread('./deploy/env1_highres.png')
        # Create a figure and set the size
    
        ax_i.imshow(img, extent=[-4,  55, -2.7, 2.8], aspect="auto")
        ax_i.grid()
        # ax_i.legend(loc='upper left', bbox_to_anchor=(1, 0.70))
        ax_i.text(0.98, 0.85, titles[i], transform=ax_i.transAxes, fontsize=12, ha='right', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
 

    fig.text(0.5, 0.05, 'Path Distance (m)', ha='center')
    fig.text(0.01, 0.5, 'Lateral Offset (m)', va='center', rotation='vertical')
    fig.legend(ax_i.get_lines(), legends, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.03))
    plt.savefig("./deploy/figures/figure10.pdf", bbox_inches='tight', dpi=1000)
    plt.show()

def plot_figure_10_animated():
    colors = ['#AB2328', '#4169E1', '#046A38']
    titles = ['BOOM + Gemmini', 'Rocket + Gemmini', 'BOOM']
    legends = ['-20°', '0°', '20°']


    gemmini_boom_files = [
        r'./deploy/hephaestus/logs/tunnel-exp-boom-gemmini-160.csv', 
        r'./deploy/hephaestus/logs/tunnel-exp-boom-gemmini-180.csv',
        r'./deploy/hephaestus/logs/tunnel-exp-boom-gemmini-200.csv'
    ]
    gemmini_boom_labels = ['BOOM + Gemmini, -20°', 'BOOM + Gemmini, 0°', 'BOOM + Gemmini, 20°']

    boom_files = [
        r'./deploy/hephaestus/logs/tunnel-exp-boom-only-160.csv', 
        r'./deploy/hephaestus/logs/tunnel-exp-boom-only-180.csv',
        r'./deploy/hephaestus/logs/tunnel-exp-boom-only-200.csv'
    ]
    boom_labels = ['BOOM, -20°', 'BOOM, 0°', 'BOOM, 20°']

    gemmini_rocket_files = [
        r'./deploy/hephaestus/logs/tunnel-exp-rocket-gemmini-160.csv', 
        r'./deploy/hephaestus/logs/tunnel-exp-rocket-gemmini-180.csv',
        r'./deploy/hephaestus/logs/tunnel-exp-rocket-gemmini-200.csv'
    ]
    gemmini_rocket_labels = ['Rocket + Gemmini, -20°','Rocket + Gemmini, 0°', 'Rocket + Gemmini, 20°']
  
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(5, 8))
    
    files = gemmini_boom_files
    labels = gemmini_boom_labels
    title = "Superscalar Core + Accelerator"

    #files = gemmini_rocket_files
    #labels = gemmini_rocket_labels
    #title = "In-Order Core + Accelerator"

    #files = boom_files
    #labels = boom_labels 
    #title = "Superscalar Core Only"


    f = plt.figure()
    f.set_figwidth(5)
    f.set_figheight(2.5)
    ax = f.add_subplot(111)
    #ax.set_position([0.13,0.2,0.8,0.7])
    ax.set_position([0.05,0.05,0.9,0.9])

    img = mpimg.imread('./deploy/env1_highres_green.png')
    plot_trajectories_2d_animated(f, ax, files, labels, colors, events=False, y_offset=-22.85, legend_loc='lower center', legend_ncols=3, legend_bbox=(0.5,-0.6), title=title, img=img, img_extent=[-4,55,-2.7,2.8], grid=False, show_axes=False)
    #ax_i.text(0.98, 0.85, titles[i], transform=ax_i.transAxes, fontsize=12, ha='right', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))


def plot_figure_11():  
    gemmini_boom_files_complex = [
        r'./deploy/hephaestus/logs/rose-hw-sw-sweep-boom-gemmini-resnet6.csv',
        r'./deploy/hephaestus/logs/rose-hw-sw-sweep-boom-gemmini-resnet11.csv',
        r'./deploy/hephaestus/logs/rose-hw-sw-sweep-boom-gemmini-resnet14.csv',
        r'./deploy/hephaestus/logs/rose-hw-sw-sweep-boom-gemmini-resnet18.csv',
        # r'./deploy/hephaestus/logs/rose-hw-sw-sweep-boom-gemmini-resnet34.csv',
  ]
    gemmini_boom_files_complex = [
        '/home/cobble/Downloads/deploy/hephaestus/logs/resnet6-fixed-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-19-05-59-39.csv',
        '/home/cobble/Downloads/deploy/hephaestus/logs/resnet11-fixed-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-19-15-21-21.csv',
        '/home/cobble/Downloads/deploy/hephaestus/logs/resnet14-fixed-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-19-01-26-01.csv',
        '/home/cobble/Downloads/deploy/hephaestus/logs/resnet18-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-17-22-47-09.csv',
        #'/home/cobble/Downloads/deploy/hephaestus/logs/resnet34-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-17-18-18-28.csv' 
        ]
    gemmini_boom_labels_complex = ['ResNet6', 'ResNet11', 'ResNet14', 'ResNet18', 'ResNet34', 'Test (ResNet14)', 'Test']
    gemmini_boom_colors_complex = ['#0033FF', '#0000FF', '#3300FF']
    gemmini_boom_colors_complex = ['#ffa921', '#f87943', '#dc535b', '#af3d69', '#783368', '#402a58']
    gemmini_boom_colors_complex = ['#AA9900', '#AB2328', '#4169E1', '#046A38', '#402A58', '#000000']
    f = plt.figure()
    f.set_figwidth(7)
    f.set_figheight(3)
    ax = plt.axes()
    
    
    # Load the image
    img = mpimg.imread('./deploy/env2_highres.png')
    # Create a figure and set the size
    
    plot_trajectories_2d(ax, gemmini_boom_files_complex, gemmini_boom_labels_complex, gemmini_boom_colors_complex, events=False, path='s-shape')
    ax.imshow(img, extent=[-4,  84, -18.55, 15.9])
    plt.title("Trajectories for DNN Achitectures")
    plt.grid()
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.45))
    plt.savefig("./deploy/figures/figure11_a.pdf", dpi=600, bbox_inches = 'tight')
    plt.show()
    
    gemmini_boom_labels_complex = ['ResNet6', 'ResNet11', 'ResNet14', 'ResNet18', 'ResNet34', 'Test (ResNet14)', 'Test']
    gemmini_boom_colors_complex = ['#0033FF', '#0000FF', '#3300FF']
    gemmini_boom_colors_complex = ['#ffa921', '#f87943', '#dc535b', '#af3d69', '#783368', '#402a58']
    gemmini_boom_colors_complex = ['#AA9900', '#AB2328', '#4169E1', '#046A38', '#402A58', '#000000']
    f = plt.figure()
    f.set_figwidth(7)
    f.set_figheight(2)
    ax = plt.axes()
    plot_time_2d(ax, gemmini_boom_files_complex, gemmini_boom_labels_complex, gemmini_boom_colors_complex, events=False, path='s-shape')
    plt.title("Lateral Position vs. Time for DNN Architectues")
    plt.grid()
    plt.legend(loc='lower center', ncols=2, bbox_to_anchor=(0.5, -0.9))
    plt.savefig("./deploy/figures/figure11_b.pdf", bbox_inches = 'tight')
    plt.show()

def plot_figure_11_animated():  
    gemmini_boom_files_complex = [
        r'./deploy/hephaestus/logs/rose-hw-sw-sweep-boom-gemmini-resnet6.csv',
        #r'./deploy/hephaestus/logs/rose-hw-sw-sweep-boom-gemmini-resnet11.csv',
        r'./deploy/hephaestus/logs/rose-hw-sw-sweep-boom-gemmini-resnet14.csv',
        r'./deploy/hephaestus/logs/rose-hw-sw-sweep-boom-gemmini-resnet18.csv',
        # r'./deploy/hephaestus/logs/rose-hw-sw-sweep-boom-gemmini-resnet34.csv',
  ]
    # gemmini_boom_files_complex = [
    #     '/home/cobble/Downloads/deploy/hephaestus/logs/resnet6-fixed-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-19-05-59-39.csv',
    #     #'/home/cobble/Downloads/deploy/hephaestus/logs/resnet11-fixed-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-19-15-21-21.csv',
    #     '/home/cobble/Downloads/deploy/hephaestus/logs/resnet14-fixed-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-19-01-26-01.csv',
    #     '/home/cobble/Downloads/deploy/hephaestus/logs/resnet18-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-17-22-47-09.csv',
    #     #'/home/cobble/Downloads/deploy/hephaestus/logs/resnet34-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-17-18-18-28.csv' 
    #     ]
    #gemmini_boom_labels_complex = ['ResNet6', 'ResNet11', 'ResNet14', 'ResNet18', 'ResNet34', 'Test (ResNet14)', 'Test']
    #gemmini_boom_colors_complex = ['#AA9900', '#AB2328', '#4169E1', '#046A38', '#402A58', '#000000']
    gemmini_boom_labels_complex = ['ResNet6',  'ResNet14', 'ResNet18', 'ResNet34', 'Test (ResNet14)', 'Test']
    gemmini_boom_colors_complex = ['#AB2328', '#4169E1', '#046A38', '#402A58', '#000000']
    f = plt.figure()
    f.set_figwidth(7)
    f.set_figheight(4)
    ax = f.add_subplot(111)
    ax.set_position([0.13,0.2,0.8,0.8])
    
    # Load the image
    img = mpimg.imread('./deploy/env2_highres_green.png')
    # Create a figure and set the size
    
    plot_trajectories_2d_animated(f, ax, gemmini_boom_files_complex, gemmini_boom_labels_complex, gemmini_boom_colors_complex, events=False, path='s-shape', legend_loc='lower center', legend_ncols=3, legend_bbox=(0.5,-0.6), title="Trajectories for DNN Architectures", img=img, img_extent=[-4,84,-18.55,15.9], grid=False, show_axes=False)
    # ax.imshow(img, extent=[-4,  84, -18.55, 15.9])
    # plt.title("Trajectories for DNN Achitectures")
    # plt.grid()
    # plt.savefig("./deploy/figures/figure11_a.pdf", dpi=600, bbox_inches = 'tight')
    # plt.show()
    
def plot_figure_test():  
    gemmini_boom_files_complex = [
        r'./logs/rose-hw-sw-sweep-boom-gemmini-resnet6.csv',
        r'./logs/rose-hw-sw-sweep-boom-gemmini-resnet11.csv',
        r'./logs/rose-hw-sw-sweep-boom-gemmini-resnet14.csv',
        r'./logs/rose-hw-sw-sweep-boom-gemmini-resnet18.csv',
        # r'./logs/rose-hw-sw-sweep-boom-gemmini-resnet34.csv',
  ]
    gemmini_boom_labels_complex = ['ResNet6', 'ResNet11', 'ResNet14', 'ResNet18', 'ResNet34', 'Test (ResNet14)', 'Test']
    gemmini_boom_colors_complex = ['#0033FF', '#0000FF', '#3300FF']
    gemmini_boom_colors_complex = ['#ffa921', '#f87943', '#dc535b', '#af3d69', '#783368', '#402a58']
    gemmini_boom_colors_complex = ['#AA9900', '#AB2328', '#4169E1', '#046A38', '#402A58', '#000000']
    f = plt.figure()
    f.set_figwidth(7)
    f.set_figheight(3)
    ax = plt.axes()
    plot_time_2d(ax, gemmini_boom_files_complex, gemmini_boom_labels_complex, gemmini_boom_colors_complex, events=False, path='s-shape')
    plt.title("")
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.45))
    plt.savefig("./figures/figurewhat.png", bbox_inches = 'tight')
    plt.show()

def plot_figure_12():  
    gemmini_boom_files = [
        #r'./named_log/gemmini-boom-s-speeds/gemmini-boom-s-velocity-6-runlog-angle-200-cycles-10000000-frames-1-2022-11-18-20-12-56.csv',
        r'./deploy/hephaestus/logs/rose-velocity-sweep-boom-gemmini-6.csv',
        r'./deploy/hephaestus/logs/rose-velocity-sweep-boom-gemmini-9.csv',
        r'./deploy/hephaestus/logs/rose-velocity-sweep-boom-gemmini-12.csv',
        ]
    gemmini_boom_labels = ['6m/s', '9m/s', '12m/s']
    gemmini_boom_colors = ['#0033FF', '#0000FF', '#3300FF']

    f = plt.figure()
    f.set_figwidth(7)
    f.set_figheight(3)
    ax = plt.axes()
    plot_trajectories_2d(ax, gemmini_boom_files, gemmini_boom_labels, ['#AB2328', '#4169E1', '#046A38'], events=False,path='s-shape')
    img = mpimg.imread('./deploy/env2_highres.png')
    # Create a figure and set the size
    
    ax.imshow(img, extent=[-4,  84, -18.55, 15.9])
    plt.title("Trajectories for Target Velocities")
    plt.grid()
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.85))
    plt.savefig("./deploy/figures/figure12_a.pdf", bbox_inches = 'tight')
    plt.show()

    f = plt.figure()
    f.set_figwidth(7)
    f.set_figheight(1.5)
    ax = plt.axes()
    plot_deadlines_2d(ax, gemmini_boom_files, gemmini_boom_labels, ['#AB2328', '#4169E1', '#046A38'], events=False)
    plt.xlabel("Distance (m)")
    plt.ylabel("Deadline (s)")
    plt.title("Deadlines for Target Velocities")
    plt.grid()
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='lower center', ncols=2, bbox_to_anchor=(0.45, -1.5))
    plt.savefig("./deploy/figures/figure12_b.pdf", bbox_inches = 'tight')
    plt.show()

    # f = plt.figure()
    # f.set_figwidth(7)
    # f.set_figheight(1.5)
    # ax = plt.axes()
    # plot_deadline_violations_2d(ax, gemmini_boom_files, gemmini_boom_labels, ['#AB2328', '#4169E1', '#046A38'], [0.085, 0.085, 0.085], events=False,path='s-shape')
    # plt.xlabel("Distance (m)")
    # plt.ylabel("Deadline Violation")
    # plt.title("Deadline Violations for Target Velocities")
    # plt.grid()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.savefig('./deploy/figures/figure14.png', bbox_inches = 'tight')
    # plt.show()
    
def plot_figure_13():  
    gemmini_boom_files_dynamic = [
        r'./deploy/hephaestus/logs/rose-hw-sw-sweep-boom-gemmini-resnet6.csv',
        r'./deploy/hephaestus/logs/rose-hw-sw-sweep-boom-gemmini-resnet11.csv',
        r'./deploy/hephaestus/logs/rose-dynamic-exp-boom-gemmini.csv',
        r'./deploy/hephaestus/logs/rose-hw-sw-sweep-boom-gemmini-resnet14.csv',     
        ]
    gemmini_boom_files_dynamic = [
        r'/home/cobble/Downloads/deploy/hephaestus/logs/resnet6-fixed-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-19-05-59-39.csv',
        r'/home/cobble/Downloads/deploy/hephaestus/logs/resnet11-fixed-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-19-15-21-21.csv',
        r'resnet14-6-complex-s-shape-low-gain-boom-gemmini-9v-10m-runlog-angle-200-cycles-20000000-frames-2-2023-02-20-21-24-42.csv',
        r'/home/cobble/Downloads/deploy/hephaestus/logs/resnet14-fixed-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-19-01-26-01.csv',     
        ]    
        
        
    gemmini_boom_labels_dynamic = ['ResNet6', 'ResNet11', 'Dynamic ResNet', 'ResNet14']
    gemmini_boom_colors_dynamic = ['#AA9900', '#AB2328', '#4169E1', '#046A38',  '#402A58', '#000000']
    latencies = [(0.06, 0.06), (0.07, 0.07), (0.06, 0.09) , (0.09, 0.09),]
    f = plt.figure()
    f.set_figwidth(7)
    f.set_figheight(3)
    ax = plt.axes()
    plot_trajectories_dynamic(ax, gemmini_boom_files_dynamic, gemmini_boom_labels_dynamic, gemmini_boom_colors_dynamic, events=False, path='s-shape')
    img = mpimg.imread('./deploy/env2_highres.png')
    # Create a figure and set the size
    
    ax.imshow(img, extent=[-4,  84, -18.55, 15.9])
    plt.title("")
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.55))
    # plt.savefig("./deploy/figures/figure13.png", bbox_inches = 'tight')
    #plt.show()

    f = plt.figure()
    f.set_figwidth(7)
    f.set_figheight(3)
    ax = plt.axes()
    plot_analysis_dynamic(ax, gemmini_boom_files_dynamic, gemmini_boom_labels_dynamic, gemmini_boom_colors_dynamic, latencies, events=False, path='s-shape')
    plt.title("")
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(0.55, 0.72))
    plt.savefig("./deploy/figures/figure13.pdf", bbox_inches = 'tight')
    plt.show()

   
def plot_figure_14():


    files = [['/home/cobble/Downloads/deploy/hephaestus/logs/resnet6-complex-s-shape-low-gain-rocket-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-21-11-03-31.csv', '/home/cobble/Downloads/deploy/hephaestus/logs/resnet6-fixed-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-19-05-59-39.csv'],['/home/cobble/Downloads/deploy/hephaestus/logs/resnet11-complex-s-shape-low-gain-rocket-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-21-10-23-05.csv', '/home/cobble/Downloads/deploy/hephaestus/logs/resnet11-fixed-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-19-15-21-21.csv'],['/home/cobble/Downloads/deploy/hephaestus/logs/resnet14-complex-s-shape-low-gain-rocket-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-21-09-43-11.csv',  '/home/cobble/Downloads/deploy/hephaestus/logs/resnet14-fixed-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-19-01-26-01.csv' ],['/home/cobble/Downloads/deploy/hephaestus/logs/resnet18-complex-s-shape-low-gain-rocket-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-21-12-45-44.csv','/home/cobble/Downloads/deploy/hephaestus/logs/resnet18-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-17-22-47-09.csv' ]]
    latencies = [[(0.090, 0.090), (0.11, 0.11), (0.13, 0.13), (0.19, 0.19), (0.30, 0.30)],
    [(0.065, 0.065), (0.07, 0.07), (0.09, 0.09), (0.13, 0.13), (0.21, 0.21)]]
    networks = ['ResNet6', 'ResNet11', 'ResNet14', 'ResNet18']
    cores = [ 'Rocket + Gemmini', 'BOOM + Gemmini']

    # files = [['/home/cobble/Downloads/deploy/hephaestus/logs/resnet6-complex-s-shape-low-gain-rocket-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-21-11-03-31.csv', '/home/cobble/Downloads/deploy/hephaestus/logs/resnet6-fixed-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-19-05-59-39.csv'],['/home/cobble/Downloads/deploy/hephaestus/logs/resnet14-complex-s-shape-low-gain-rocket-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-21-09-43-11.csv',  '/home/cobble/Downloads/deploy/hephaestus/logs/resnet14-fixed-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-19-01-26-01.csv' ],['/home/cobble/Downloads/deploy/hephaestus/logs/resnet18-complex-s-shape-low-gain-rocket-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-21-12-45-44.csv','/home/cobble/Downloads/deploy/hephaestus/logs/resnet18-complex-s-shape-low-gain-boom-gemmini-9v-runlog-angle-200-cycles-20000000-frames-2-2023-02-17-22-47-09.csv' ]]
    # latencies = [[(0.090, 0.090), (0.13, 0.13), (0.19, 0.19), (0.30, 0.30)],
    # [(0.06, 0.06), (0.09, 0.09), (0.13, 0.13), (0.21, 0.21)]]
    # networks = ['ResNet6','ResNet14', 'ResNet18']
    # cores = [ 'Rocket + Gemmini', 'BOOM + Gemmini']
    plot_hw_sw_sweep_bars(files,networks, cores, latencies)

def plot_figure_15():
    files = [r'./deploy/hephaestus/logs/rose-perf-tunnel-rocket-gemmini-10_000_000.csv',
             r'./deploy/hephaestus/logs/rose-perf-tunnel-rocket-gemmini-20_000_000.csv',
             r'./deploy/hephaestus/logs/rose-perf-tunnel-rocket-gemmini-100_000_000.csv',
             r'./deploy/hephaestus/logs/rose-perf-tunnel-rocket-gemmini-400_000_000.csv'
             ]
             
    files = [ '/home/cobble/Downloads/deploy/hephaestus/logs/gemmini-rocket-maze-runlog-angle-200-cycles-10000000-frames-1-2022-11-07-09-34-06.csv',
              '/home/cobble/Downloads/deploy/hephaestus/logs/gemmini-rocket-maze-runlog-angle-200-cycles-20000000-frames-2-2022-11-07-08-44-25.csv',
              '/home/cobble/Downloads/deploy/hephaestus/logs/gemmini-rocket-maze-runlog-angle-200-cycles-30000000-frames-3-2022-11-07-10-17-10.csv',
              '/home/cobble/Downloads/deploy/hephaestus/logs/gemmini-rocket-maze-runlog-angle-200-cycles-50000000-frames-5-2022-11-07-10-52-36.csv',
              '/home/cobble/Downloads/deploy/hephaestus/logs/gemmini-rocket-maze-runlog-angle-200-cycles-100000000-frames-10-2022-11-07-12-05-03.csv',
              '/home/cobble/Downloads/deploy/hephaestus/logs/gemmini-rocket-maze-runlog-angle-200-cycles-200000000-frames-20-2022-11-07-14-23-57.csv',
              '/home/cobble/Downloads/deploy/hephaestus/logs/gemmini-rocket-maze-runlog-angle-200-cycles-400000000-frames-40-2022-11-08-10-29-17.csv']
    # indices = [10_000_000]
    indices = [10_000_000, 20_000_000, 30_000_000, 50_000_000, 100_000_000, 200_000_000, 400_000_000]
    plot_rollouts_roofline(files, indices)

def plot_figure_16():
    gemmini_rocket_sweep_files = [
       r'./deploy/hephaestus/logs/rose-perf-tunnel-rocket-gemmini-10_000_000.csv',
             r'./deploy/hephaestus/logs/rose-perf-tunnel-rocket-gemmini-20_000_000.csv',
             r'./deploy/hephaestus/logs/rose-perf-tunnel-rocket-gemmini-100_000_000.csv',
             #'/home/cobble/Downloads/deploy/hephaestus/logs/gemmini-rocket-maze-runlog-angle-200-cycles-400000000-frames-40-2022-11-08-10-29-17.csv',
             r'./deploy/hephaestus/logs/rose-perf-tunnel-rocket-gemmini-400_000_000.csv'
             ]
    gemmini_rocket_sweep_colors = ['#ffa921', '#f87943', '#dc535b', '#af3d69']
    gemmini_rocket_sweep_colors = [
        '#0036ff', 
        '#ae00d7', 
        '#e700a5', 
        '#ff0071'
        ]
    gemmini_rocket_sweep_colors = [
        None, 
        None, 
        None, 
        None, 
        ]

    indices = [1, 2, 10, 40] 
    mses = calculate_mses(gemmini_rocket_sweep_files, indices)

    gemmini_rocket_sweep_labels = [
        f'10M Cycles', 
        f'20M Cycles  (MSE: {round(mses[1])})',  
        f'100M Cycles (MSE: {round(mses[2])})', 
        f'400M Cycles (MSE: {round(mses[3])})'
        ]
        
    gemmini_rocket_sweep_labels = [
        f'10M Cycles', 
        f'20M Cycles  (MSE: {round(mses[1])})',  
        f'100M Cycles (MSE: {round(mses[2])})', 
        f'400M Cycles (MSE: {round(mses[3])})'
        ]
    f = plt.figure()
    f.set_figwidth(7)
    f.set_figheight(4)
    ax = plt.axes()
    xs = [1, 6]
    ys = [-1.2, 0.2]
    rect = patches.Rectangle((xs[0], ys[0]), xs[1]-xs[0], ys[1]-ys[0], linewidth=2, linestyle='--', edgecolor='gray', facecolor='none')
    ax.add_patch(rect)
    plot_trajectories_2d(ax, gemmini_rocket_sweep_files, gemmini_rocket_sweep_labels, gemmini_rocket_sweep_colors, events=False,  y_offset = -22.85, show_time=False)
    img = mpimg.imread('./deploy/env1_highres.png')
    # Create a figure and set the size
    
    ax.imshow(img, extent=[-4,  55, -2.7, 2.8], aspect="auto")
    # ax.set_ylim(-2, 4)
    # ax.set_xlim(0, 5)
    # ax.set_ylim(-1, 0.25)
    
    
    
    plt.xlabel("Path Distance (m)")
    plt.ylabel("Lateral Offset (m)")
    #plt.title("Trajectories vs Synchronization Granularity for RoSE Simulations (Rocket + Gemmini)")
    #ax.set_yticks(np.arange(-1.5, 1.5, 0.5 ), minor=True)
    plt.grid(which='both')
    plt.legend(loc="upper center", bbox_to_anchor=(0.5,0.97), ncols=2, fontsize=11)
    # ax.set_zlabel("Altitude (m)")
    plt.savefig("./deploy/figures/figure16_a.pdf", bbox_inches = 'tight', dpi=1000)
    plt.show()
    
    f = plt.figure()
    f.set_figwidth(4)
    f.set_figheight(2.5)
    ax = plt.axes()
    plot_trajectories_2d(ax, gemmini_rocket_sweep_files, gemmini_rocket_sweep_labels, gemmini_rocket_sweep_colors, events=True,  y_offset = -22.85, show_time=False, show_legend=False)
    ax.set_xlim(xs[0], xs[1])
    # ax.set_xlim(0, 5)
    ax.set_ylim(ys[0], ys[1])

    #plt.xlabel("Distance (m)")
    #plt.ylabel("Lateral Offset (m)")
    #plt.title("Trajectories vs Synchronization Granularity for RoSE Simulations (Rocket + Gemmini)")
    plt.grid()
    #plt.legend(bbox_to_anchor=(1,1), ncols=2, fontsize=12)
    # ax.set_zlabel("Altitude (m)")
    plt.savefig("./deploy/figures/figure16_b.pdf", bbox_inches = 'tight')
    plt.show()

def plot_figure_bw():  
    gemmini_boom_files_complex = [
        r'./deploy/hephaestus/logs/rose-bw-sweep-boom-gemmini-0.csv',
        r'./deploy/hephaestus/logs/rose-bw-sweep-boom-gemmini-8192.csv',
        r'./deploy/hephaestus/logs/rose-bw-sweep-boom-gemmini-32768.csv'
  ]
    gemmini_boom_labels_complex = ['0', '8192', '32768']
    gemmini_boom_labels_complex = ['Unlimited BW', '500 KB/s', '125 KB/s']
    gemmini_boom_colors_complex = ['#0033FF', '#0000FF', '#3300FF']
    gemmini_boom_colors_complex = ['#ffa921', '#f87943', '#dc535b', '#af3d69', '#783368', '#402a58']
    gemmini_boom_colors_complex = ['#AA9900', '#AB2328', '#4169E1', '#046A38', '#402A58', '#000000']
    f = plt.figure()
    f.set_figwidth(7)
    f.set_figheight(3)
    ax = plt.axes()
    
    
    # Load the image
    img = mpimg.imread('./deploy/env2_highres_green.png')
    # Create a figure and set the size
    
    plot_trajectories_2d(ax, gemmini_boom_files_complex, gemmini_boom_labels_complex, gemmini_boom_colors_complex, events=False, path='s-shape')
    ax.imshow(img, extent=[-4,  84, -18.55, 15.9])
    plt.grid()
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.45))
    plt.savefig("./deploy/figures/figurebw.png", dpi=600, bbox_inches = 'tight')
    plt.show()

    f = plt.figure()
    f.set_figwidth(7)
    f.set_figheight(2)
    ax = plt.axes()
    plot_time_2d(ax, gemmini_boom_files_complex, gemmini_boom_labels_complex, gemmini_boom_colors_complex, events=False, path='s-shape')
    plt.grid()
    plt.xlim([0,15])
    plt.legend(loc='lower center', ncols=3, bbox_to_anchor=(0.5, -0.7))
    plt.savefig("./deploy/figures/figurebw_lateral.png", bbox_inches = 'tight')
    plt.show()
    
if __name__ == "__main__":
    # plt.rcParams["text.usetex"] = True
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.size'] = 14
    #plot_figure_10()
    #plot_figure_10_animated()
    #plot_figure_11_animated()
    #plot_figure_12()
    #plot_figure_13()
    #plot_figure_14()
    #plot_figure_16()
    #plot_figure_15()
    plot_figure_bw()
