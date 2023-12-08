import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_animation(states, length=100):
    fig, ax = plt.subplots()

    # 初始化第一帧
    img = ax.imshow(states[0], cmap='gray', interpolation='none')

    def animate(i):
        img.set_data(states[i])
        return img,

    ani = animation.FuncAnimation(fig, animate, frames=length, interval=50, blit=True)
    plt.close()  # 防止显示多余的静态图像

    # 保存动画
    ani.save('ising_model_evolution.mp4', fps=60)


def horizontal_color_func(value):
    # 将 value 映射从 [-1, 1] 到 [0, 1]    
    normalized_value = (value + 1) / 2
    # 红色到绿色的线性插值
    return (float(1 - normalized_value.item()), float(normalized_value.item()), 0)

def vertical_color_func(value):
    # 将 value 映射从 [-1, 1] 到 [0, 1]
    normalized_value = (value + 1) / 2
    # 红色到绿色的线性插值
    return (float(1 - normalized_value.item()), float(normalized_value.item()), 0)

def create_grid_with_colored_borders_alone(d, h_borders, v_borders):
    fig, ax = plt.subplots()

    # 绘制横线
    for i in range(d):
        for j in range(d):
            plt.plot([j, j + 1], [d - i - 1, d - i - 1], color=horizontal_color_func(h_borders[i, j]), lw=1)

    # 绘制竖线
    for i in range(d-1):
        for j in range(d-1):
            plt.plot([j+1, j+1], [d - i - 2, d - i - 1], color=vertical_color_func(v_borders[i, j]), lw=1)

    ax.set_xlim(0, d)
    ax.set_ylim(-0.1, d-1)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()

def create_grid_with_colored_borders(ax, d, h_borders, v_borders):
    # 绘制横线
    for i in range(d):
        for j in range(d):
            ax.plot([j, j + 1], [d - i - 1, d - i - 1], color=horizontal_color_func(h_borders[i, j]), lw=3, zorder=2)

    # 绘制竖线
    for i in range(d-1):
        for j in range(d-1):
            ax.plot([j + 1, j + 1], [d - i - 2, d - i - 1], color=vertical_color_func(v_borders[i, j]), lw=3, zorder=2)

    ax.set_xlim(0, d)
    ax.set_ylim(0, d-1)
    ax.set_aspect('equal')
    ax.axis('off')

def create_spin_animation(d, h_borders, v_borders, states, filename):
    fig, ax = plt.subplots(dpi=300)

    # 使用自定义函数绘制边框
    create_grid_with_colored_borders(ax, d, h_borders, v_borders)

    # 设置初始状态和坐标范围
    img = ax.imshow(states[0], cmap='gray', interpolation='none', extent=[0, d, 0, d-1], zorder=1)

    def update_frame(num, states, img):
        img.set_data(states[num])
        return img,

    # 创建动画
    ani = animation.FuncAnimation(fig, update_frame, frames=len(states), fargs=(states, img), interval=50, blit=True)

    # 保存为 MP4
    ani.save(filename, writer='ffmpeg', fps=60, dpi=300, bitrate=6000)

    plt.close(fig)



