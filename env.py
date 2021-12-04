# import numpy as np
# class envModel() :
#     def __init__(self):
#         self.action_space_n = 4
#         pass
#
#     def reset(self):
#         pass
#
#     def take_action(self, action):
#         pass
#
#     def get_state(self):
#         old_state = np.random.rand(1, 80, 80, 4)
#         action = 1
#         reward = 0
#         #new_state = np.random.rand(1, 80, 80, 4)
#         #new_state = old_state
#         new_state = old_state.copy()
#         is_terminal = 1
#
#         return old_state, action, reward, new_state, is_terminal
#
#     def close(self):
#         print("end")
import math
import time

import gym
from gym import spaces
import numpy as np
import cv2
from config import Config
MAX_SHIP_NUM = 2
SHIP_TYPE_SELF = 0
SHIP_TYPE_OTHER = 1
VIEWPORT_W = 1000
VIEWPORT_H = 1000
ENCOUNTER = 1
CROSS_LEFT = 2
CROSS_RIGHT = 3
OVERTAKE = 4
encounter_type = OVERTAKE
class envModel(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.viewer = None ##??
        self.action_space_n = 3
        self.action_space = spaces.Discrete(360)  ##??
        self.observation_space = spaces.Box(low=0, high=VIEWPORT_H, shape=(MAX_SHIP_NUM, 4), dtype=np.float32)  #??

        self.scale = 0.1
        parser = Config.parser
        self.args = parser.parse_args()
        self.args.input_shape = tuple(self.args.input_shape)
        self.action = 0

        self.d_min = 20
        self.reset()
        from gym.envs.classic_control import rendering
        self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)

    def reset(self):
        # 管理所有船的列表， reset时先清空
        self.ships = []

        # # 随机生成MAX_SHIP_NUM - 1个其它船
        # for i in range(MAX_SHIP_NUM - 1):
        #     self.ships.append(self.randship(SHIP_TYPE_OTHER, encounter_type))

        # 生成agent
        self.selfship = self.randship(SHIP_TYPE_SELF, encounter_type)

        # 把agent加入管理列表
        self.ships.append(self.selfship)

        self.goal = self.randgoal()
        # 更新观察数据
        self.state = np.vstack([ship.state for (_, ship) in enumerate(self.ships)])

        self.d = math.sqrt((self.selfship.state[0] - self.goal[0]) ** 2 + (self.selfship.state[1] - self.goal[1]) ** 2)
        self.d_last = self.d
        self.is_terminal = False
        self.new_state = None
        # 返回
        return self.goal

    #@staticmethod
    def randship(self,_t: np.int,type):
        # _t 为船类型（agent或其它）
        # Ship class 参数为坐标x,y,类型_t
        # VIEWPORT_W, VIEWPORT_H为地图宽高
        #_b = Ship(np.random.rand(1)[0] * VIEWPORT_W, np.random.rand(1)[0] * VIEWPORT_H, _t,type)
        if _t == SHIP_TYPE_SELF:
            _b = Ship(5000,1000,self.args.self_speed,self.args.self_heading, _t, type)
        else:
            _b = Ship(np.random.rand(1)[0] * VIEWPORT_W, np.random.rand(1)[0] * VIEWPORT_H, _t,type)
        return _b

    def randgoal(self):
        goal = []
        x = 5000
        y = 9500
        goal.append(x)
        goal.append(y)
        return goal

    def turn(self,action):
        return 3 * (action - 1)
    def headingToAngle(self,angle):
        result = 90 - angle
        if result < -180:
            result += 360
        return result / 180 * math.pi
    def step(self, action):
        reward = 0.0  # 奖励初始值为0
        done = False  # 该局游戏是否结束
        for _, ship in enumerate(self.ships):
            if(ship.type == SHIP_TYPE_OTHER):
                ship.last_state = ship.state[:]#x,y,v,w(弧度),yaw(方位角 度数),vx,vy ##计算角与方位角
                ship.state[0] = ship.state[0] + ship.state[2] * math.cos(self.headingToAngle(ship.state[4]))
                ship.state[1] = ship.state[1] + ship.state[2] * math.sin(self.headingToAngle(ship.state[4]))
            else:
                #ship.y += 5
                #ship.state[1] = ship.y
                deta_heading = self.turn(action)
                ship.last_state = ship.state[:]
                ship.state[3] = deta_heading / 180 * math.pi
                if ship.state[3] != 0:
                    ship.state[0] = ship.last_state[0] + (
                            math.sin(ship.state[3] + self.headingToAngle(ship.last_state[4])) - math.sin(self.headingToAngle(ship.last_state[4]))) * \
                                         ship.state[2] / ship.state[3]
                    ship.state[1] = ship.last_state[1] - (
                            math.cos(ship.state[3] + self.headingToAngle(ship.last_state[4])) - math.cos(self.headingToAngle(ship.last_state[4]))) * \
                                         ship.state[2] / ship.state[3]
                else:
                    ship.state[0] = ship.last_state[0] + ship.state[2] * math.cos(self.headingToAngle(ship.state[4]))
                    ship.state[1] = ship.last_state[1] + ship.state[2] * math.sin(self.headingToAngle(ship.state[4]))
                ship.state[4] = ship.last_state[4] + deta_heading
                ship.state[5] = ship.state[2] * math.cos(ship.state[4])
                ship.state[6] = ship.state[2] * math.sin(ship.state[4])

        self.state = np.vstack([ship.state for (_, ship) in enumerate(self.ships)])
        self.d_last = self.d
        self.d = math.sqrt((self.selfship.state[0] - self.goal[0]) ** 2 + (self.selfship.state[1] - self.goal[1]) ** 2)


    def getreward(self):
        reward = 0
        reward_list = []
        reward_list.append(self.d)
        # r1假如上一时刻到目标的距离<这一时刻到目标的距离就会有负的奖励
        if self.d_last < self.d:
            reward = reward - 0.005 * (self.d - self.d_last)
            reward_list.append(reward)
        if self.d_last >= self.d:
            reward = reward + 0.005 * (self.d_last - self.d)
            reward_list.append(reward)
        #r2 角速度变化即角度变化
        r2 = - 5 * abs(self.selfship.state[3] - self.selfship.last_state[3])
        reward_list.append(r2)
        reward = reward + r2
        #r3右转向
        r3 = 0.01
        if self.action == 0:
            reward_list.append(0)
        elif self.selfship.last_state[4] > 0 and self.selfship.state[4] < 0:
            # reward += -r3
            reward_list.append(0)
        elif self.selfship.last_state[4] < 0 and self.selfship.state[4] > 0:
            reward += r3
            reward_list.append(r3)
        else:
            if self.selfship.state[4] > self.selfship.last_state[4]:
                # reward -= r3
                reward_list.append(0)
            else:
                reward += r3
                reward_list.append(r3)

        # 到达目标点有正的奖励
        if self.d < self.d_min:
            reward_list.append(20)
            reward = reward + 20
            print("Get 20 reward------goal point!!!!!!")
            self.is_terminal = True

        # if self.is_terminal == False:
        #     for i in range(1):
        #         if not self.ship.judge_point(self.obs_pos[i][0], self.obs_pos[i][1]):
        #             reward_list.append(-5)
        #             print("Obstacle!!!!!")
        #             self.done_list = True  # 终止
        #             break
        #         else:
        #             self.done_list = False
        #         if not self.ship.judge_point(self.obs_robot_state[0][0], self.obs_robot_state[0][1]):
        #             reward_list.append(-5)
        #             print("Obstacle!!!!!")
        #             self.ship.show()
        #             self.done_list = True  # 终止
        #             break
        #         else:
        #             self.done_list = False

        return reward, reward_list, self.is_terminal
        # 与动态障碍发生碰撞
        # 动态障碍为边长0.8m的正方体

    def get_state(self):

        if self.viewer != None:
            array = self.render(mode='rgb_array')
        else:
            array = np.zeros((1000,1000,3))
        #array = np.zeros((1000, 1000, 3))
        #old_state = np.random.rand(1, 80, 80, 4)

        # input1-->相机
        # shape of image_matrix [768,1024,3]
        image_matrix = np.uint8(array)
        image_matrix = cv2.resize(image_matrix, (self.args.input_shape[0], self.args.input_shape[1]))
        # shape of image_matrix [80,80,3]
        result_image = image_matrix[:]
        image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_RGB2GRAY)
        # shape of image_matrix [80,80]
        # image_matrix = np.reshape(image_matrix, (self.args.input_shape[0], self.args.input_shape[1]))

        # print("shape of image matrix={}".format(self.image_matrix.shape))

        self.old_state = self.new_state[:] if self.new_state is not None else None
        self.new_state = image_matrix.reshape((1, 80, 80, 1))
        reward,sub_reward,_ = self.getreward()
        # new_state = np.random.rand(1, 80, 80, 4)
        # new_state = old_state

        return self.old_state, self.action, reward, self.new_state, self.is_terminal, sub_reward, self.selfship.state[0], self.selfship.state[1],self.selfship.state[4]

    def render(self, mode='human'):
        # 按照gym的方式创建一个viewer, 使用self.scale控制缩放大小
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W , VIEWPORT_H )

        # 渲染所有的船
        for item in self.state:
            # 从状态中获取坐标、分数、类型
            _x, _y, _s, _t = item[0] * self.scale, item[1] * self.scale, item[2], item[3]

            # transform用于控制物体位置、缩放等
            transform = rendering.Transform()
            transform.set_translation(_x , _y )

            # 添加一个⚪，来表示船
            # 中心点: (x, y)
            # 半径:
            # 颜色: 其它船为蓝色、agent船为红/紫色
            self.viewer.draw_circle(math.sqrt(10 / math.pi), 30, color=(0, 0, 255)).add_attr(transform)

        # 然后直接渲染（没有考虑性能）
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


    def close(self):
        pass

class Ship():
    def __init__(self, x: np.float32, y: np.float32,speed: np.float32,heading: np.float32, t: np.int, type: np.int):
        '''
            x   初始x坐标
            y   初始y坐标
            s   初始分
            w	移动方向，弧度值
            t   船类型
        '''

        self.speed = speed
        self.heading = heading
        # self.s = score
        # self.w = way * 2 * math.pi / 360.0  # 角度转弧度
        self.type = t

        self.id = None  # 生成船唯一id
        self.lastupdate = time.time()  # 上一次的计算时间
        self.timescale = 100  # 时间缩放，或者速度的缩放

        self.state = [x, y, speed, 0.0, heading, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
        self.last_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.roll_state = [0.0, 0.0, 0.0]
        self.encounter_type = type





    def state(self):
        return [self.x, self.y, self.s, self.t]


if __name__ == '__main__':
    env = envModel()

    while True:
        env.step(0)
        #env.render()
