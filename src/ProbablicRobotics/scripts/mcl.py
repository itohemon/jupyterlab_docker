#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../scripts/")
from robot import *
from scipy.stats import multivariate_normal
import random
import copy


# In[2]:


class Particle:
    def __init__(self, init_pose, weight):
        self.pose = init_pose
        self.weight = weight

    def motion_update(self, nu, omega, time, noise_rate_pdf):
        ns = noise_rate_pdf.rvs() # 順にnn, no, on, oo
        # 速度に雑音を加える
        noised_nu = nu + ns[0]*math.sqrt(abs(nu)/time) + ns[1]*math.sqrt(abs(omega)/time)
        # 角速度に雑音を加える
        noised_omega = omega + ns[2]*math.sqrt(abs(nu)/time) + ns[3]*math.sqrt(abs(omega)/time)
        # 姿勢を更新する
        self.pose = IdealRobot.state_transition(noised_nu, noised_omega, time, self.pose)

    def observation_update(self, observation, envmap, distance_dev_rate, direction_dev):
        # それぞれのランドマークの観測結果から一つの観測結果をdとして取り出す
        for d in observation:
            # dをセンサ値とランドマークのIDに分解
            obs_pos = d[0]
            obs_id = d[1]

            # パーティクルの位置と地図からランドマークの距離と角度を算出（極座標に変換）
            pos_on_map = envmap.landmarks[obs_id].pos # 観測したランドマークのIDから地図中の当該のランドマークの位置を読み込む
            particle_suggest_pos = IdealCamera.observation_function(self.pose, pos_on_map) # 極座標に変換を行う

            # 尤度の計算
            distance_dev = distance_dev_rate * particle_suggest_pos[0]
            cov = np.diag(np.array([distance_dev**2, direction_dev ** 2]))
            self.weight *= multivariate_normal(mean=particle_suggest_pos, cov=cov).pdf(obs_pos)


# In[3]:


class Mcl:
    def __init__(self, envmap, init_pose, num, motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
                distance_dev_rate=0.14, direction_dev=0.05):
        self.particles = [Particle(init_pose, 1.0/num) for i in range(num)]
        self.map = envmap
        self.distance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev
        self.ml = self.particles[0]
        self.pose = self.ml.pose

        v = motion_noise_stds # 標準偏差σabに対応

        # ４次元のガウス分布のオブジェクトを作る
        c = np.diag([v["nn"]**2, v["no"]**2, v["on"]**2, v["oo"]**2]) # 対角行列を作って返す
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)

    def set_ml(self):
        i = np.argmax([p.weight for p in self.particles])
        self.ml = self.particles[i]
        self.pose = self.ml.pose
    
    def motion_update(self, nu, omega, time):
        for p in self.particles: p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)
                                                             

    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation, self.map, self.distance_dev_rate, self.direction_dev)
        self.set_ml()
        self.resampling()

    def resampling(self):
        ws = np.cumsum([e.weight for e in self.particles]) # 重みを累積して足していく（最後の要素が重みの合計になる）
        if ws[-1] < 1e-100: ws = [e + 1e-100 for e in ws] # 重みの合計が０のときの処理

        step = ws[-1] / len(self.particles) # 正規化されていない場合はステップが「重みの合計値/N」になる
        r = np.random.uniform(0.0, step)
        cur_pos = 0
        ps = []

        while (len(ps) < len(self.particles)):
            if r < ws[cur_pos]:
                ps.append(self.particles[cur_pos]) # もしかしたらcur_posがはみ出るかもしれないが例外処理は割愛
                r += step
            else:
                cur_pos += 1
        
        self.particles = [copy.deepcopy(e) for e in ps] # 選んだリストからパーティクルを取り出し重みを均一にする
        for p in self.particles: p.weight = 1.0 / len(self.particles) # 重みの正規化
            
    def draw(self, ax, elems):
        # 全パーティクルのX座標、Y座標をリスト化する
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]

        # パーティクルの向きを描画するために、向きをベクトルで表したときのX座標、Y座標成分のリストを作る
        # 重みに比例した長さの矢印でパーティクルを描く
        # パーティクルの数が多いほど１個あたりの重みの平均値が小さくなるのでパーティクルの数も掛ける
        vxs = [math.cos(p.pose[2]) * p.weight * len(self.particles) for p in self.particles]
        vys = [math.sin(p.pose[2]) * p.weight * len(self.particles) for p in self.particles]
        
        elems.append(ax.quiver(xs, ys, vxs, vys,\
                               angles='xy', scale_units='xy', scale=1.5, color="blue", alpha=0.5))


# In[4]:


class EstimationAgent(Agent):
    def __init__(self, time_interval, nu, omega, estimator):
        super().__init__(nu, omega)
        self.estimator = estimator
        self.time_interval = time_interval

        self.prev_nu = 0.0
        self.prev_omega = 0.0

    def decision(self, observation=None):
        # 一つ前の制御指令値でパーティクルの姿勢を更新する
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        self.estimator.observation_update(observation)
        return self.nu, self.omega
    
    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)
        x, y, t = self.estimator.pose
        s = "({:.2f}, {:.2f}, {})".format(x, y, int(t*180/math.pi) % 360)
        elems.append(ax.text(x, y+0.1, s, fontsize=8))


# In[ ]:




