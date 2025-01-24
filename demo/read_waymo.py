import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_waymo_data(pkl_path):
    # 加载pkl文件
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def visualize_scene(data):
    # 提取地图特征
    map_features = data['map_features']
    
    plt.figure(figsize=(10, 10))
    
    # 绘制道路
    for feature in map_features.values():
        if feature['type'] == 'LANE_SURFACE_STREET':
            polyline = np.array(feature['polyline'])
            plt.plot(polyline[:, 0], polyline[:, 1], 'b-', alpha=0.5)
    
    # 打印并绘制车辆轨迹
    print("\n查找车辆轨迹...")
    colors = plt.cm.rainbow(np.linspace(0, 1, len(data['tracks'])))  # 为每个轨迹生成不同颜色
    
    for (track_id, track), color in zip(data['tracks'].items(), colors):
        track_type = track.get('type', track['metadata'].get('type', 'NA'))
        if track_type == 'VEHICLE':
            print(f"轨迹ID: {track_id}")
            print(f"type: {track_type}")
            print("其他信息:")
            for key, value in track['metadata'].items():
                print(f"  {key}: {value}")
            
            # 打印并绘制轨迹坐标
            if 'state' in track:
                state = track['state']
                valid_mask = state['valid']  # 获取有效数据的掩码
                positions = state['position'][valid_mask]  # 只使用有效的位置数据
                
                if len(positions) > 0:
                    print("轨迹坐标 (x, y, z):")
                    for pos in positions:
                        print(f"  ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                    
                    # 绘制轨迹
                    plt.plot(positions[:, 0], positions[:, 1], '-', color=color, label=f'Vehicle {track_id}')
                    plt.plot(positions[0, 0], positions[0, 1], 'o', color=color)  # 标记起点
                    plt.plot(positions[-1, 0], positions[-1, 1], 's', color=color)  # 标记终点
            
            print("-" * 20)
    
    plt.title('车辆轨迹图')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    
    plt.axis('equal')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 加载数据
    pkl_path = "/home/hui/rl/cat/raw_scenes_500/298.pkl"  # 替换为实际路径
    data = load_waymo_data(pkl_path)
    
    # 打印基本信息
    print("数据集信息:")
    print(f"场景ID: {data['metadata']['scenario_id']}")
    print(f"地图特征数量: {len(data['map_features'])}")
    print(f"车辆轨迹数量: {len(data['tracks'])}")
    
    # 可视化场景
    visualize_scene(data)
