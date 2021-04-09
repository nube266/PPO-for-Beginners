import torch
from torch.distributions import MultivariateNormal
from network import FeedForwardNN


class PPO:
    def __init__(self, env):
        # ハイパーパラメータの初期化
        self._init_hyperparameters()

        # 環境の情報の抽出
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # ALG STEP 1
        # ActorとCriticのネットワークの初期化
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        # 共分散行列の作成
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def learn(self, total_timesteps):
        t_so_far = 0  # これまでにシミュレートされたタイムステップ
        while t_so_far < total_timesteps:  # ALG STEP 2
            # ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

    def rollout(self):
        # バッチデータ
        batch_obs = []             # 観測値:(observations-バッチごとのタイムステップ数、観測値の次元）
        batch_acts = []            # アクション:(actions-バッチあたりのタイムステップ数、アクションの次元）
        batch_log_probs = []       # 対数確率:(log probabilities-バッチあたりのタイムステップ数）
        batch_rews = []            # 報酬:(rewards-エピソードの数、エピソードごとのタイムステップの数）
        batch_rtgs = []            # 報酬:(reward to go's-バッチあたりのタイムステップ数）
        batch_lens = []            # バッチの長さ:(batch length-エピソードの数）

        # このバッチでこれまでに実行されたタイムステップの数
        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []
            obs = self.env.reset()
            done = False
            for ep_t in range(self.max_timesteps_per_episode):
                # これまでのバッチを実行したタイムステップを増やす
                t += 1
                # 観測データの取得
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)
                # 報酬、アクション、対数確率を取得
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                # 終了判定
                if done:
                    break
            # エピソードの長さと報酬を取得
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # データを指定された形状のテンソルに再形成してから返す
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # ALG STEP 4
        batch_rtgs = self.compute_rtgs(batch_rews)
        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, obs):
        # 平均的なアクションのためのネットワーク(アクター)
        # self.actor.forward(obs)を呼び出すのと同じ
        mean = self.actor(obs)
        # 多変量正規分布を定義
        dist = MultivariateNormal(mean, self.cov_mat)
        # 分布からアクションをサンプリングして、その対数確率を取得する
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # サンプリングされたアクションと、そのアクションのlog probを返す
        return action.detach().numpy, log_prob.detach()

    def compute_rtgs(self, batch_rews):
        # バッチごと・エピソードごとのリワード（rtg）を返す
        batch_rtgs = []
        # 各エピソードを逆方向に反復し、batch_rtgsで同じ順序を維持
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0  # これまでの割引報酬
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # rewards-to-goをテンソルに変換
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800  # バッチあたりのタイムステップ
        self.max_timesteps_per_episode = 1600  # エピソードあたりのタイムステップ
        self.gamma = 0.95  # 割引率
