import math
import time

import gym
from gym import spaces
import numpy as np
import cv2
from config import Config
from gym.envs.classic_control import rendering

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
        self.action_space_n = 15
        #self.action_space = spaces.Discrete(360)  ##??
        #self.observation_space = spaces.Box(low=0, high=VIEWPORT_H, shape=(MAX_SHIP_NUM, 4), dtype=np.float32)  #??

        self.scale = 0.1
        parser = Config.parser
        self.args = parser.parse_args()
        self.args.input_shape = tuple(self.args.input_shape)
        self.action = 0

        self.d_min = 20
        self.reset()
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

        self.rel_angle = self.getRelAngle()
        self.rel_angle_last = self.rel_angle
        self.is_terminal = False
        self.new_state = None
        # 返回
        return self.goal

    def getRelAngle(self):
        angle = abs(math.atan2((self.goal[1] - self.selfship.state[1]),
                       (self.goal[0] - self.selfship.state[0]))) / math.pi * 180
        heading = self.angleToHeading(self.selfship.state[4])
        result = angle - heading#目标船在本船右方为正
        return result
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
        y = 6500
        goal.append(x)
        goal.append(y)
        return goal

    def turn(self,action):
        # return 3 * (action - 1)
        T = 50
        K = 0.05
        Te = 2.5
        rudder_order = action
        self.selfship.last_roll_state = self.selfship.roll_state[:]
        self.selfship.roll_state[0] = self.selfship.last_roll_state[0] + self.selfship.last_roll_state[1]  #角度
        self.selfship.roll_state[1] = self.selfship.last_roll_state[1] - self.selfship.last_roll_state[1] / T + K / T * self.selfship.last_roll_state[2]
        # self.roll_state[2] = self.last_roll_state[2] - self.last_roll_state[2] / Te + rudder_order /Te
        self.selfship.roll_state[2] = rudder_order
        return self.selfship.last_roll_state[1]

    def headingToAngle(self,angle):
        result = 90 - angle
        if result < -180:
            result += 360
        return result / 180 * math.pi

    def angleToHeading(self,angle):
        result = 90 - angle
        if result < 0:
            result += 360
        return result

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
        self.rel_angle_last = self.rel_angle
        self.rel_angle = self.getRelAngle()
        return self.selfship.roll_state[:]

    def getreward(self):
        reward = 0
        reward_list = []
        reward_list.append(self.d)
        # r1假如上一时刻到目标的距离<这一时刻到目标的距离就会有负的奖励
        r1 = -0.005 * (self.d - self.d_last) * 2 * 2 / 10
        reward += r1
        reward_list.append(r1)
        # r2 目标船相对本船的方位
        r2_value = 0.2 / 10
        r2_symble = 0
        if abs(self.rel_angle) < 0.1:
            r2_symble = 1
        elif abs(self.rel_angle_last) < 0.1:
            r2_symble = -1
        elif self.rel_angle > 0 and self.rel_angle_last > 0:
            if self.rel_angle - self.rel_angle_last > 0 :
                r2_symble = -1
            else:
                r2_symble = 1
        elif self.rel_angle > 0 and self.rel_angle_last < 0:
            r2_symble = -1
        elif self.rel_angle < 0 and self.rel_angle_last > 0:
            r2_symble = -1
        elif self.rel_angle < 0 and self.rel_angle_last < 0:
            if self.rel_angle - self.rel_angle_last > 0 :
                r2_symble = 1
            else:
                r2_symble = -1
        else:
            print("未知错误")
        r2 = r2_value * r2_symble
        reward += r2
        reward_list.append(r2)

        # r3 角速度变化即角度变化
        r3 = - 5 * abs(self.selfship.state[3] - self.selfship.last_state[3]) / 10
        reward_list.append(r3)
        reward = reward + r3
        #r4右转向
        r4 = 0.01 / 10
        if self.action == 0:
            reward_list.append(0)
        elif self.selfship.last_state[4] > 0 and self.selfship.state[4] < 0:
            reward += -r4
            reward_list.append(0)
        elif self.selfship.last_state[4] < 0 and self.selfship.state[4] > 0:
            # reward += r4
            # reward_list.append(r4)
            pass
        else:
            if self.selfship.state[4] > self.selfship.last_state[4]:
                reward -= r4
                reward_list.append(0)
            else:
                # reward += r4
                # reward_list.append(r4)
                pass
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

    def test_state(self,state):
        total = 0
        list = state.tolist()
        for i in range(len(list)):
            for j in range(len(list[i])):
                for m in range(len(list[i][j])):
                    if list[i][j][m] != 255.0:
                        #print(i, j, m)
                        total += 1
        print("total point num is：", total)

    def input_initialization(self, env_info):
        state = env_info[0]  # laser info + self state
        state_set = []
        for i in range(self.Num_skipFrame * self.Num_stackFrame):
            state_set.append(state)
        state_stack = np.zeros((self.Num_stackFrame, self.Num_dataSize))
        for stack_frame in range(self.Num_stackFrame):
            state_stack[(self.Num_stackFrame - 1) - stack_frame,
            :] = state_set[-1 - (self.Num_skipFrame * stack_frame)]

        observation = env_info[1]  # image info
        observation_set = []
        for i in range(self.Num_skipFrame * self.Num_stackFrame):
            observation_set.append(observation)
        # 使用观察组根据跳帧和堆叠帧的数量堆叠帧
        observation_stack = np.zeros(
            (self.img_size, self.img_size, self.Num_stackFrame))
        # print("shape of observation stack={}".format(observation_stack.shape))
        for stack_frame in range(self.Num_stackFrame):
            observation_stack[:, :, stack_frame] = observation_set[-1 -
                                                                   (self.Num_skipFrame * stack_frame)]
        observation_stack = np.uint8(observation_stack)

        return observation_stack, observation_set, state_stack, state_set

    def test_state1(self, state):
        total = 0
        list = state.tolist()
        for i in range(len(list)):
            for j in range(len(list[i])):
                if list[i][j] != 255.0:
                    # print(i, j)
                    total += 1
        print("total point num is：", total)
        return total

    def test_state2(self, state):
        total = 0
        list = state.tolist()
        for i in range(len(list)):
            for j in range(len(list[i])):
                for m in range(len(list[i][j])):
                    if list[i][j][m][0] != 255.0:
                        # print(i, j, m)
                        total += 1
        print("total point num is：", total)
    def get_state(self):
        if self.args.is_cnn == 1:
            if self.viewer != None:
                array = self.render(mode='rgb_array')
            else:
                array = np.zeros((1000,1000,3))
            #array = np.zeros((1000, 1000, 3))
            #old_state = np.random.rand(1, 80, 80, 4)


            # shape of image_matrix [1000,1000,3]
            #self.test_state(array)
            image_matrix = np.uint8(array)
            # image_matrix = cv2.resize(image_matrix, (self.args.input_shape[0], self.args.input_shape[1]))
            # # shape of image_matrix [80,80,3]
            # result_image = image_matrix[:]
            # self.test_state(result_image)
            image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_RGB2GRAY)
            image_matrix = cv2.resize(image_matrix, (self.args.input_shape[0], self.args.input_shape[1]))
            #self.test_state1(image_matrix1)
            # shape of image_matrix [80,80]
            # image_matrix = np.reshape(image_matrix, (self.args.input_shape[0], self.args.input_shape[1]))

            # print("shape of image matrix={}".format(self.image_matrix.shape))

            self.old_state = self.new_state[:] if self.new_state is not None else None
            self.new_state = image_matrix.reshape((1, self.args.input_shape[0], self.args.input_shape[1], 1))
        else:
            state = []
            state.append(self.rel_angle)
            state.append(self.rel_angle_last)
            state.append(self.selfship.roll_state[2])
            state.append(self.selfship.last_roll_state[2])
            self.old_state = self.new_state[:] if self.new_state is not None else None
            self.new_state = np.array(state).reshape((1,self.args.lstm_input_length,1))
        #self.test_state2(self.new_state)
        reward, sub_reward, is_terminal= self.getreward()
        # new_state = np.random.rand(1, 80, 80, 4)
        # new_state = old_state

        return self.old_state, self.action, reward, self.new_state, is_terminal, sub_reward, self.selfship.state[0], self.selfship.state[1],self.selfship.state[4]

    def render(self, mode='human'):
        # 按照gym的方式创建一个viewer, 使用self.scale控制缩放大小

        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W , VIEWPORT_H )

        # 渲染所有的船
        for item in self.state:
            # 从状态中获取坐标、分数、类型
            _x, _y, _s, _t = item[0] * self.scale, item[1] * self.scale, item[2], item[3]

            # # transform用于控制物体位置、缩放等
            # transform = rendering.Transform()
            # transform.set_translation(_x , _y )
            #
            # # 添加一个⚪，来表示船
            # # 中心点: (x, y)
            # # 半径:
            # # 颜色: 其它船为蓝色、agent船为红/紫色
            # self.viewer.draw_circle(10, 30, color=(0, 0, 255)).add_attr(transform)
            self.draw_circle(_x, _y, 5, (0, 0, 255))
            # self.draw_circle(500, 500, 500, (0, 0, 255))
        # 然后直接渲染（没有考虑性能）
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def draw_circle(self, x, y, r, c):
        assert isinstance(self.viewer, rendering.Viewer)

        transform = rendering.Transform()
        transform.set_translation(x, y)
        self.viewer.draw_circle(r, 30, color=c).add_attr(transform)

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

        # self.id = None  # 生成船唯一id
        # self.lastupdate = time.time()  # 上一次的计算时间
        # self.timescale = 100  # 时间缩放，或者速度的缩放

        self.state = [x, y, speed, 0.0, heading, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
        self.last_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.roll_state = [0.0, 0.0, 0.0] #舵令、角速度
        self.last_roll_state = [0.0, 0.0, 0.0]
        self.encounter_type = type





    def state(self):
        return [self.x, self.y, self.s, self.t]


if __name__ == '__main__':
    env = envModel()

    while True:
        env.step(0)
        #env.render()
