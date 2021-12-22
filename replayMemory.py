import numpy as np
import random
from sumTree import SumTree

class ReplayMemory():

    def __init__(self,max_size,window_size,input_shape,lstm_input_length):

        self._max_size = max_size
        self._window_size = window_size
        self._WIDTH = input_shape[0]
        self._HEIGHT = input_shape[1]
        self._memory = []
        self._lstm_input_length = lstm_input_length


    def append(self,old_state,action,reward,new_state,is_terminal):

        num_sample = len(old_state)
        if len(self._memory) >= self._max_size:
            del(self._memory[0:num_sample])

        self._memory.append((old_state,action,reward,new_state,is_terminal))

    def sample(self,batch_size, is_cnn, indexes=None):

        samples = random.sample(self._memory,min(batch_size,len(self._memory)))
        zipped = list(zip(*samples))
        if is_cnn == 1:
            zipped[0] = np.reshape(zipped[0],(-1,self._WIDTH,self._HEIGHT,self._window_size))
            zipped[3] = np.reshape(zipped[3],(-1,self._WIDTH,self._HEIGHT,self._window_size))
        else:
            zipped[0] = np.reshape(zipped[0], (-1, self._lstm_input_length, self._window_size))
            zipped[3] = np.reshape(zipped[3], (-1, self._lstm_input_length, self._window_size))
        return zipped


class PriorityExperienceReplay():

    def __init__(self,max_size,
                 window_size,
                 input_shape,
                 lstm_input_length):
        self.tree = SumTree(max_size)
        self._max_size = max_size
        self._window_size = window_size
        self._WIDTH = input_shape[0]
        self._HEIGHT = input_shape[1]
        self._lstm_input_length = lstm_input_length
        self.e = 0.01
        self.a = 0.6


    def _getPriority(self,error):
        return (error + self.e) ** self.a

    def append(self,old_state,action,reward,new_state,is_terminal):

        p = self._getPriority(0.5)
        self.tree.add(p,data=(old_state, action,reward,new_state,is_terminal))

    def sample(self,batch_size, is_cnn, indexes=None):
        data_batch = []
        idx_batch = []
        p_batch = []
        segment = self.tree.total_and_count()[0] / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a,b)
            (idx,p,data) = self.tree.get(s)
            data_batch.append(data)
            idx_batch.append(idx)
            p_batch.append(p)

        zipped = list(zip(*data_batch))

        if is_cnn == 1:
            zipped[0] = np.reshape(zipped[0],(-1,self._WIDTH,self._HEIGHT,self._window_size))
            zipped[3] = np.reshape(zipped[3],(-1,self._WIDTH,self._HEIGHT,self._window_size))
        else:
            zipped[0] = np.reshape(zipped[0], (-1, self._lstm_input_length, self._window_size))
            zipped[3] = np.reshape(zipped[3], (-1, self._lstm_input_length, self._window_size))
        sum_p,count = self.tree.total_and_count()

        return zipped,idx_batch,p_batch,sum_p,count

    def update(self,idx_list,error_list):
        for idx,error in zip(idx_list,error_list):
            p = self._getPriority(error)
            self.tree.update(idx,p)


