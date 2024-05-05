#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
matplotlib.use('nbagg')
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np
from IPython.display import HTML


# In[2]:


class World:
    def __init__(self, time_span, time_interval, debug=False):
        self.objects = [] # ここにロボットなどのオブジェクトを登録
        self.debug = debug
        self.time_span = time_span # 何秒間シミュレーションをするか
        self.time_interval = time_interval # delta-tを指定する
        
    def append(self, obj):
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(8, 8)) # 8x8インチの図を準備
        ax = fig.add_subplot(111) # サブプロットを準備
        ax.set_aspect('equal') # 縦横比を座標の値と一致させる
        ax.set_xlim((-5, 5)) # X軸を-5m x 5m の範囲で描画
        ax.set_ylim((-5, 5)) # Y軸も同様
        ax.set_xlabel("X", fontsize=20)
        ax.set_ylabel("Y", fontsize=20)

        elems = []

        if self.debug:
            for i in range(1000):
                self.one_step(i, elems, ax) # デバッグ時はアニメーションさせない
        else:
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax),
                                         frames=int(self.time_span / self.time_interval) + 1, interval=int(self.time_interval * 1000), repeat=False)
            return HTML(self.ani.to_jshtml())

    def one_step(self, i, elems, ax):
        while elems: elems.pop().remove() # 二重の描画を防ぐため
        time_str = "t = %.2f[s]" % (self.time_interval * i)
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(self.time_interval)


# In[3]:


class IdealRobot:
    def __init__(self, pose, agent=None, sensor=None, color="black"):
        self.pose = pose # 引数から姿勢の初期値を設定
        self.r = 0.2 # これは描画のためなので固定値
        self.color = color # 引数から描画するときの色を設定
        self.agent = agent
        self.poses = [pose] # 軌跡の描画用
        self.sensor = sensor

    def draw(self, ax, elems):
        x, y, theta = self.pose # 姿勢の変数を分解して３つの変数へ
        xn = x + self.r * math.cos(theta) # ロボットの鼻先のX座標
        yn = y + self.r * math.sin(theta) # ロボットの鼻先のY座標
        elems += ax.plot([x, xn], [y, yn], color=self.color) # ロボットの向きを示す線分の描画
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c)) # 上のpatches.Circleでロボットの胴体を示す円を作ってサププロットへ登録     

        self.poses.append(self.pose)
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black")

        if self.sensor and len(self.poses) > 1:
            self.sensor.draw(ax, elems, self.poses[-2])

        if self.agent and hasattr(self.agent, "draw"):
            self.agent.draw(ax, elems)
    
    @classmethod
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10: # 角速度がほぼゼロとそうでない場合で場合分け
            return pose + np.array([nu * math.cos(t0),
                                    nu * math.sin(t0),
                                    omega]) * time
        else:
            return pose + np.array([nu / omega * ( math.sin(t0 + omega * time) - math.sin(t0)),
                                    nu / omega * (-math.cos(t0 + omega * time) + math.cos(t0)),
                                    omega * time])

    def one_step(self, time_interval):
        if not self.agent: return
        
        # センサがついていたらセンサで観測を行う。センサがついていなかったら何もしない
        obs = self.sensor.data(self.pose) if self.sensor else None
        
        nu, omega = self.agent.decision(obs)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)


# In[4]:


class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def decision(self, observation=None):
        return self.nu, self.omega


# In[5]:


class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = None

    def draw(self, ax, elems):
        # scatter()は散布図に点を打つためのメソッド
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))


# In[6]:


class Map:
    def __init__(self):
        self.landmarks = [] # 空のランドマークのリストを準備

    def append_landmark(self, landmark): # ランドマークを追加
        landmark.id = len(self.landmarks) # 追加するランドマークにIDを与える
        self.landmarks.append(landmark)

    def draw(self, ax, elems): # 描画(Landmarkのdrawを順に呼び出す)
        for lm in self.landmarks: lm.draw(ax, elems)


# In[7]:


class IdealCamera:
    def __init__(self, env_map, distance_range=(0.5, 6.0), direction_range=(-math.pi/3, math.pi/3)):
        self.map = env_map
        self.lastdata = []

        self.distance_range = distance_range
        self.direction_range = direction_range

    def visible(self, polarpos): # ランドマークが観測できる条件
        if polarpos is None:
            return False

        return self.distance_range[0] <= polarpos[0] <= self.distance_range[1] \
            and self.direction_range[0] <= polarpos[1] <= self.direction_range[1]
    
    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            if self.visible(z):
                observed.append((z, lm.id)) # ロボットから見たランドマークの極座標とIDをタプルにしてobsevedリストに追加する

        self.lastdata = observed
        return observed

    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[0:2]
        # phiはロボットから見たランドマークの向き
        phi = math.atan2(diff[1], diff[0]) - cam_pose[2]
        while phi >= np.pi: phi -= 2*np.pi
        while phi < -np.pi: phi += 2*np.pi

        # hypot(x, y)は原点から(x,y)までの距離を返す関数
        return np.array([np.hypot(*diff), phi]).T # 戻り値は距離と角度

    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            # World座標系でのランドマークの位置を計算
            x, y, theta = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * math.cos(direction + theta)
            ly = y + distance * math.sin(direction + theta)
            # ロボットランドマークの間にピンクの線分を引く
            elems += ax.plot([x, lx], [y, ly], color="pink")


# In[9]:


if __name__ == '__main__':
    world = World(30, 0.1)
    
    # 地図を生成して３つのランドマークを追加
    m = Map()
    m.append_landmark(Landmark(2, -2))
    m.append_landmark(Landmark(-1, 3))
    m.append_landmark(Landmark(3, 3))
    
    world.append(m) # Worldに地図を登録
    
    straight = Agent(0.2, 0.0) # 0.2[m/s]で直進
    circling = Agent(0.2, 10.0/180*math.pi) # 0.2[m/s],10[rad/s](円を描く)
    
    robot1 = IdealRobot( np.array([2, 3, math.pi/6]).T, sensor=IdealCamera(m), agent=straight) # ロボットのインスタンス生成（色を省略）
    robot2 = IdealRobot( np.array([-2, -1, math.pi/5*6]).T, sensor=IdealCamera(m), agent=circling, color="red") # ロボットのインスタンスを生成（色を指定）
    
    world.append(robot1)
    world.append(robot2)
    
    world.draw()

