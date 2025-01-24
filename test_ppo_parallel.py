import ray
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import numpy as np
"""
批量并行测试

原始:

"""
@ray.remote
class TestWorker:
    def __init__(self, checkpoint_path, worker_id, num_episodes):
        self.checkpoint_path = checkpoint_path
        self.worker_id = worker_id
        self.num_episodes = num_episodes
        
        # 配置测试环境
        self.env_config = {
            "use_render": False,  # 并行测试时关闭渲染
            "data_directory": "/home/hui/rl/cat/scenes/turn_scenes",
            "num_scenarios": 100,
            "start_scenario_index": 0,
            "vehicle_config": {
                "lidar": {
                    "num_lasers": 30,
                    "distance": 50,
                    "num_others": 3
                },
                "side_detector": {"num_lasers": 30},
                "lane_line_detector": {"num_lasers": 12},
            }
        }
        
        register_env('waymo-env', self.env_creator)
        
        # 初始化指标
        self.arr_num = 0
        self.fail_arr_num = 0 
        self.total_reward = 0
        self.total_route_completion = 0
        self.total_crashes = 0
        self.total_out_of_road = 0

        config = (
            PPOConfig()
            .environment("waymo-env")
            .framework("torch")
            .rollouts(num_rollout_workers=0)
            .training()
        )
        
        self.model = config.build()
        self.model.restore(checkpoint_path)
        
    def env_creator(self, env_config):
        return WaymoEnv(self.env_config)
        
    def setup(self):
        from metadrive.engine.engine_utils import close_engine
        close_engine()
        self.env = WaymoEnv(self.env_config)
        self.obs = self.env.reset()
        
    def run(self):
        self.setup()
        cur_episode = 0
        
        while cur_episode < self.num_episodes:
            action = self.model.compute_single_action(
                self.obs,
                explore=False
            )
            
            obs, reward, done, info = self.env.step(action)
            
            if done:
                if info.get('arrive_dest'):
                    self.arr_num += 1
                else:
                    self.fail_arr_num += 1
                    
                self.total_route_completion += info['route_completion'] 
                self.total_crashes += float(info['crash_vehicle'])
                self.total_out_of_road += float(info['out_of_road'])
                
                print(f"Worker {self.worker_id} - Episode {cur_episode + 1}/{self.num_episodes}")
                
                self.obs = self.env.reset()
                cur_episode += 1
            
            self.obs = obs
            self.total_reward += reward
            
        self.env.close()
        
        # 返回此worker的结果
        return {
            'success': self.arr_num,
            'route_completion': self.total_route_completion,
            'crashes': self.total_crashes,
            'out_of_road': self.total_out_of_road,
            'total_reward': self.total_reward,
            'episodes': self.num_episodes
        }

def run_parallel_test(checkpoint_path, num_workers=4, episodes_per_worker=25):
    """
    使用多个worker并行运行测试
    
    Args:
        checkpoint_path: 模型检查点路径
        num_workers: worker数量
        episodes_per_worker: 每个worker测试的回合数
    """
    ray.init()
    
    # 创建workers
    workers = [
        TestWorker.remote(
            checkpoint_path=checkpoint_path,
            worker_id=i,
            num_episodes=episodes_per_worker
        )
        for i in range(num_workers)
    ]
    
    # 并行执行测试
    results = ray.get([worker.run.remote() for worker in workers])
    
    # 汇总结果
    total_episodes = sum(r['episodes'] for r in results)
    total_success = sum(r['success'] for r in results)
    total_route_completion = sum(r['route_completion'] for r in results)
    total_crashes = sum(r['crashes'] for r in results)
    total_reward = sum(r['total_reward'] for r in results)
    
    # 打印最终结果
    print("\n=== Final Results ===")
    print(f"Total Episodes: {total_episodes}")
    print(f"Success Rate: {total_success/total_episodes:.3f}")
    print(f"Average Route Completion: {total_route_completion/total_episodes:.3f}")
    print(f"Crash Rate: {total_crashes/total_episodes:.3f}")
    print(f"Average Reward: {total_reward/total_episodes:.3f}")
    
    ray.shutdown()

if __name__ == "__main__":
    checkpoint_path = '/home/hui/rl/cat/models/waymo_ppo_final_2025_01_24_15_09/checkpoint_000200'
    import time
    start_time = time.time()
    # 运行并行测试
    test_episode_num = 1000
    num_workers = 15 # 根据配置而定
    run_parallel_test(
        checkpoint_path=checkpoint_path,
        num_workers=num_workers,
        episodes_per_worker=int(test_episode_num/num_workers) + 1
    )
    end_time = time.time()
    print(f'Cost time: {end_time - start_time} seconds')