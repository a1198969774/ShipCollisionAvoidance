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
encounter_type = CROSS_RIGHT
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

        self.d_min = 200

        self.obs_num = 0

        self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.reset()


    def type_encounter(agent_x, agent_y, agent_angle):
        k1 = math.tan(agent_angle)
        b1 = agent_y - k1 * agent_x
        if agent_angle < 0:
            obs_angle = agent_angle - math.pi
        else:
            obs_angle = agent_angle + math.pi
        while (True):
            deta_angle = np.random.normal(0, 0.5) * math.pi / 2
            if deta_angle > math.pi * (-5) / 180 and deta_angle < math.pi * 5 / 180:
                break
        obs_angle += deta_angle
        if obs_angle > math.pi:
            obs_angle -= math.pi * 2
        if obs_angle < -math.pi:
            obs_angle += math.pi * 2
        k2 = math.tan(obs_angle)
        b2 = agent_y - k2 * agent_x
        goal_x = agent_x + math.cos(agent_angle) * VIEWPORT_W
        goal_y = agent_y + math.sin(agent_angle) * VIEWPORT_W
        k3 = k2
        b3 = goal_y - k3 * goal_x
        if b2 > b3:
            max = b2
            min = b3
        else:
            max = b3
            min = b2
        deta_b = max - min
        while (True):
            rand_ratio = np.random.normal(0.5, 0.5)
            if rand_ratio > 0.2 and rand_ratio < 0.8:
                break
        b4 = min + rand_ratio * deta_b
        k4 = k2
        x = (b4 - b1) / (k1 - k4)
        y = k1 * x + b1
        test = (x - agent_x) / math.cos(agent_angle)
        distance_agent = math.sqrt((x - agent_x) ** 2 + (y - agent_y) ** 2)
        obs_speed = 5
        distance_start = distance_agent / 6 * obs_speed
        obs_x = x - distance_start * math.cos(obs_angle)
        obs_y = y - distance_start * math.sin(obs_angle)
        # plt.plot([obs_x, x], [obs_y, y])
        return obs_x, obs_y, obs_angle, obs_speed

    def type_overtake(agent_x, agent_y, agent_angle):
        k1 = math.tan(agent_angle)
        b1 = agent_y - k1 * agent_x
        obs_angle = agent_angle
        while (True):
            deta_angle = np.random.normal(0, 0.1) * math.pi / 2
            if deta_angle > math.pi * (-67.5) / 180 and deta_angle < math.pi * 67.5 / 180:
                break
        obs_angle += deta_angle
        if obs_angle > math.pi:
            obs_angle -= math.pi * 2
        if obs_angle < -math.pi:
            obs_angle += math.pi * 2
        k2 = math.tan(obs_angle)
        b2 = agent_y - k2 * agent_x
        # goal_x = agent_x + math.cos(agent_angle) * VIEWPORT_W
        # goal_y = agent_y + math.sin(agent_angle) * VIEWPORT_W
        goal_x = 5000
        goal_y = 6500
        k3 = k2
        b3 = goal_y - k3 * goal_x
        if b2 > b3:
            max = b2
            min = b3
        else:
            max = b3
            min = b2
        deta_b = max - min
        while (True):
            rand_ratio = np.random.normal(0.5, 0.5)
            if rand_ratio > 0.5 and rand_ratio < 0.8:
                break
        b4 = min + rand_ratio * deta_b
        k4 = k2
        x = (b4 - b1) / (k1 - k4)
        y = k4 * x + b4
        test = (x - agent_x) / math.cos(agent_angle)
        distance_agent = math.sqrt((x - agent_x) ** 2 + (y - agent_y) ** 2)
        obs_speed = 3.5
        distance_start = distance_agent / 6.618 * obs_speed
        obs_x = x - distance_start * math.cos(obs_angle)
        obs_y = y - distance_start * math.sin(obs_angle)
        # plt.plot([obs_x, x], [obs_y, y])
        return obs_x, obs_y, obs_angle / math.pi * 180, obs_speed

    def type_cross_left(agent_x, agent_y, agent_angle):
        k1 = math.tan(agent_angle)
        b1 = agent_y - k1 * agent_x
        if agent_angle < -math.pi / 2:
            obs_angle = agent_angle + 3 * math.pi / 2
        else:
            obs_angle = agent_angle - math.pi / 2
        while (True):
            deta_angle = np.random.normal(-0.3375, 0.5) * math.pi / 2
            if deta_angle > math.pi * (-85) / 180 and deta_angle < math.pi * 22.5 / 180:
                break
        obs_angle += deta_angle
        if obs_angle > math.pi:
            obs_angle -= math.pi * 2
        if obs_angle < -math.pi:
            obs_angle += math.pi * 2
        k2 = math.tan(obs_angle)
        b2 = agent_y - k2 * agent_x
        goal_x = agent_x + math.cos(agent_angle) * VIEWPORT_W
        goal_y = agent_y + math.sin(agent_angle) * VIEWPORT_W
        k3 = k2
        b3 = goal_y - k3 * goal_x
        if b2 > b3:
            max = b2
            min = b3
        else:
            max = b3
            min = b2
        deta_b = max - min
        while (True):
            rand_ratio = np.random.normal(0.5, 0.5)
            if rand_ratio > 0.2 and rand_ratio < 0.5:
                break
        b4 = min + rand_ratio * deta_b
        k4 = k2
        x = (b4 - b1) / (k1 - k4)
        y = k1 * x + b1
        test = (x - agent_x) / math.cos(agent_angle)
        distance_agent = math.sqrt((x - agent_x) ** 2 + (y - agent_y) ** 2)
        obs_speed = 5
        distance_start = distance_agent / 6 * obs_speed
        obs_x = x - distance_start * math.cos(obs_angle)
        obs_y = y - distance_start * math.sin(obs_angle)
        # plt.plot([obs_x, x], [obs_y, y])
        return obs_x, obs_y, obs_angle, obs_speed

    def type_cross_right(agent_x, agent_y, agent_angle):
        k1 = math.tan(agent_angle)
        b1 = agent_y - k1 * agent_x
        if agent_angle > math.pi / 2:
            obs_angle = agent_angle - 3 * math.pi / 2
        else:
            obs_angle = agent_angle + math.pi / 2
        while (True):
            deta_angle = np.random.normal(0.3375, 0.5) * math.pi / 2
            if deta_angle > math.pi * (-22.5) / 180 and deta_angle < math.pi * 85 / 180:
                # print(deta_angle)
                break
        obs_angle += deta_angle
        if obs_angle > math.pi:
            obs_angle -= math.pi * 2
        if obs_angle < -math.pi:
            obs_angle += math.pi * 2
        k2 = math.tan(obs_angle)
        b2 = agent_y - k2 * agent_x
        goal_x = 5000
        goal_y = 6500
        k3 = k2
        b3 = goal_y - k3 * goal_x
        if b2 > b3:
            max = b2
            min = b3
        else:
            max = b3
            min = b2
        deta_b = max - min
        while (True):
            rand_ratio = np.random.normal(0.5, 0.5)
            if rand_ratio > 0.5 and rand_ratio < 0.8:
                break
        b4 = min + rand_ratio * deta_b
        k4 = k2
        x = (b4 - b1) / (k1 - k4)
        y = k4 * x + b4
        test = (x - agent_x) / math.cos(agent_angle)
        distance_agent = math.sqrt((x - agent_x) ** 2 + (y - agent_y) ** 2)
        obs_speed = 5.14
        distance_start = distance_agent / 6.168 * obs_speed
        obs_x = x - distance_start * math.cos(obs_angle)
        obs_y = y - distance_start * math.sin(obs_angle)
        # plt.plot([obs_x, x], [obs_y, y])
        return obs_x, obs_y, obs_angle / math.pi * 180, obs_speed

    switch = {'encounter': type_encounter,  # 注意此处不要加括号
              'overtake': type_overtake,
              'cross_left': type_cross_left,
              'cross_right': type_cross_right,
              }

    def reset(self):
        # 管理所有船的列表， reset时先清空
        # if self.viewer is not None:
        #     del self.viewer
        #
        self.viewer.close()
        self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)

        self.ships = []
        self.obs_list = []
        # # 随机生成MAX_SHIP_NUM - 1个其它船
        # for i in range(MAX_SHIP_NUM - 1):
        #     self.ships.append(self.randship(SHIP_TYPE_OTHER, encounter_type))

        # 生成agent
        self.selfship = self.randship(SHIP_TYPE_SELF, encounter_type)
        self.ships.append(self.selfship)

        self.obship1 = self.randship(SHIP_TYPE_OTHER, encounter_type)
        self.ships.append(self.obship1)

        self.goal = self.randgoal()

        self.randobs(self.obs_num)
        # 更新观察数据
        self.state = np.vstack([ship.state for (_, ship) in enumerate(self.ships)])

        self.d = math.sqrt((self.selfship.state[0] - self.goal[0]) ** 2 + (self.selfship.state[1] - self.goal[1]) ** 2)
        self.d_last = self.d

        self.rel_angle = self.getRelAngle()
        self.rel_angle_last = self.rel_angle

        self.rel_obs1_angle = self.getObs1RelAngle()
        self.rel_obs1_angle_last = self.rel_obs1_angle

        self.is_terminal = False
        self.new_state = None
        # 返回
        return self.goal

    def randobs(self, num):
        obs = [] # x,y,r,t
        while num != 0:
            type_num = np.random.randint(2) # 0: circle 1: square
            r_num = np.random.randint(200, 400)
            if type_num == 1:
                r_num *= 2
            x_num = np.random.randint(r_num * 3, 10000 - r_num * 6)
            y_num = np.random.randint(r_num * 2, 10000 - r_num * 2)
            if type_num == 0 and math.sqrt((5000 - x_num) ** 2 + (
                            6500 - y_num) ** 2) < r_num:
                pass
            elif type_num == 1 and 6500 < y_num and 6500 > y_num - r_num and 5000 > x_num and 5000 < x_num + r_num:
                pass
            elif type_num == 0 and math.sqrt((5000 - x_num) ** 2 + (
                            1000 - y_num) ** 2) < r_num:
                pass
            elif type_num == 1 and 1000 < y_num and 1000 > y_num - r_num and 5000 > x_num and 5000 < x_num + r_num:
                pass
            else:
                obs.append(x_num)
                obs.append(y_num)
                obs.append(r_num)
                obs.append("circle" if type_num == 0 else "square")
                self.obs_list.append(obs)
                num -= 1
                obs = []
    def getRelAngle(self):
        angle = abs(math.atan2((self.goal[1] - self.selfship.state[1]),
                       (self.goal[0] - self.selfship.state[0]))) / math.pi * 180
        heading = self.angleToHeading(self.selfship.state[4])
        result = angle - heading#目标船在本船右方为正
        return result

    def getObs1RelAngle(self):
        angle = abs(math.atan2((self.obship1.state[1] - self.selfship.state[1]),
                       (self.obship1.state[0] - self.selfship.state[0]))) / math.pi * 180
        heading = self.angleToHeading(self.selfship.state[4])
        result = angle - heading#目标船在本船右方为正
        return result

    #@staticmethod
    def randship(self,ship_type: np.int,type):
        # _t 为船类型（agent或其它）
        # Ship class 参数为坐标x,y,类型_t
        # VIEWPORT_W, VIEWPORT_H为地图宽高
        #_b = Ship(np.random.rand(1)[0] * VIEWPORT_W, np.random.rand(1)[0] * VIEWPORT_H, _t,type)
        if ship_type == SHIP_TYPE_SELF:
            _b = Ship(5000, 1000, self.args.self_speed, self.args.self_heading, ship_type, type)
        else:
            obs1_x, obs1_y, obs1_angle, obs1_speed = self.switch.get("cross_right")(5000, 1000, math.pi / 2)
            _b = Ship(obs1_x, obs1_y, self.args.self_speed, self.angleToHeading(obs1_angle), ship_type,type)
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
                self.rel_obs1_angle_last = self.rel_obs1_angle
                self.rel_obs1_angle = self.getObs1RelAngle()
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
        r3 = - 5 * abs(self.selfship.state[3] - self.selfship.last_state[3]) * 10
        reward_list.append(r3)
        reward = reward + r3
        #r4右转向
        r4 = 0.01
        if self.action == 0:
            reward_list.append(0)
        elif self.selfship.last_state[4] > 0 and self.selfship.state[4] < 0:
            reward += -r4
            reward_list.append(-r4)
        elif self.selfship.last_state[4] < 0 and self.selfship.state[4] > 0:
            # reward += r4
            # reward_list.append(r4)
            pass
        else:
            if self.selfship.state[4] > self.selfship.last_state[4]:
                reward -= r4
                reward_list.append(-r4)
            else:
                # reward += r4
                # reward_list.append(r4)
                pass


        if math.sqrt((self.selfship.state[0] - self.obship1.state[0]) ** 2 +(self.selfship.state[1] - self.obship1.state[1]) ** 2) < 555.6:
            reward_list.append(-10)
            reward = reward - 10
            print("Get -10 reward------obstacle")
            self.is_terminal = True

        for item in self.obs_list:
            _x, _y, _r, _t = item[0], item[1], item[2], item[3]
            if _t == "circle":
                if math.sqrt((self.selfship.state[0] - _x) ** 2 +(self.selfship.state[1] - _y) ** 2) < _r:
                    reward_list.append(-10)
                    reward = reward - 10
                    print("Get -10 reward------obstacle")
                    self.is_terminal = True
            if _t == "square":
                # 上 下 左 右
                y1 = _y
                y2 = _y - _r
                x3 = _x
                x4 = _x + _r
                if self.selfship.state[1] <= y1 and self.selfship.state[1] >= y2 and self.selfship.state[0] > x3 and self.selfship.state[0] < x4:
                    reward_list.append(-10)
                    reward = reward - 10
                    print("Get -10 reward------obstacle")
                    self.is_terminal = True
        # 到达目标点有正的奖励
        if self.d < self.d_min:
            reward_list.append(20)
            reward = reward + 20
            print("Get 20 reward------goal point!!!!!!")
            self.is_terminal = True



        return reward, reward_list, self.is_terminal


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
            # self.new_state = image_matrix.reshape((1, self.args.input_shape[0], self.args.input_shape[1], 1))
            self.new_state = image_matrix
        elif self.args.is_cnn == 0:
            state = []
            state.append(self.rel_angle)
            state.append(self.rel_angle_last)
            state.append(self.selfship.roll_state[2])
            state.append(self.selfship.last_roll_state[2])
            state.append(self.rel_obs1_angle)
            state.append(self.rel_obs1_angle_last)
            self.old_state = self.new_state[:] if self.new_state is not None else None
            # self.new_state = np.array(state).reshape((1,self.args.lstm_input_length,1))
            self.new_state = state
        #self.test_state2(self.new_state)
        else:
            state = []
            state1 = []
            state1.append(self.rel_angle)
            state1.append(self.rel_angle_last)
            state1.append(self.selfship.roll_state[2])
            state1.append(self.selfship.last_roll_state[2])
            state1.append(self.rel_obs1_angle)
            state1.append(self.rel_obs1_angle_last)
            state.append(state1)

            if self.viewer != None:
                array = self.render(mode='rgb_array')
            else:
                array = np.zeros((1000,1000,3))
            image_matrix = np.uint8(array)
            image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_RGB2GRAY)
            image_matrix = cv2.resize(image_matrix, (self.args.input_shape[0], self.args.input_shape[1]))
            state.append(image_matrix)

            self.old_state = self.new_state[:] if self.new_state is not None else None
            self.new_state = state
        reward, sub_reward, is_terminal= self.getreward()
        # new_state = np.random.rand(1, 80, 80, 4)
        # new_state = old_state

        return self.action, reward, self.new_state, is_terminal, sub_reward, self.selfship.state[0], self.selfship.state[1],self.selfship.state[4], self.obship1.state[0], self.obship1.state[1]

    def render(self, mode='human'):
        # 按照gym的方式创建一个viewer, 使用self.scale控制缩放大小

        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W , VIEWPORT_H )


        for ship in self.ships:
            if ship.type == SHIP_TYPE_SELF:
                self.draw_square(ship.state[0] * self.scale, ship.state[1] * self.scale, 50 * self.scale, (0, 0, 255))
            else:
                self.draw_circle(ship.state[0] * self.scale, ship.state[1] * self.scale, 50 * self.scale, (0, 0, 0))


        for item in self.obs_list:
            _x, _y, _r, _t = item[0], item[1], item[2], item[3]
            if _t == "circle":
                self.draw_circle(_x * self.scale, _y * self.scale, _r * self.scale, (0, 0, 0))
            if _t == "square":
                self.draw_square(_x * self.scale, _y * self.scale, _r * self.scale, (0, 0, 0))
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def draw_circle(self, x, y, r, c):
        # 画一个半径为 的圆
        circle = rendering.make_circle(r)
        # 添加一个平移操作
        circle_transform = rendering.Transform(translation=(x, y))
        # 让圆添加平移这个属性
        circle.add_attr(circle_transform)
        circle.set_color(c[0],c[1],c[2])
        self.viewer.add_geom(circle)

    def draw_square(self, x, y, r, c):
        # 画一个矩形 左下 右下 右上 左上
        square = rendering.make_polygon([(x , y - r), (x + r, y - r), (x + r, y), (x, y)], True)
        # 添加一个平移操作
        square_transform = rendering.Transform(translation=(0, 0))
        # 让圆添加平移这个属性
        square.add_attr(square_transform)
        square.set_color(c[0],c[1],c[2])
        self.viewer.add_geom(square)

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




if __name__ == '__main__':
    env = envModel()

    while True:
        env.step(0)
        #env.render()
