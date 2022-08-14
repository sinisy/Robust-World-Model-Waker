import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import common
import expl


class Agent(common.Module):

  def __init__(self, config, obs_space, act_space, step, domain):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.tfstep = tf.Variable(int(self.step), tf.int64)
    self.wm = WorldModel(config, obs_space, self.tfstep, domain)
    if domain in common.DOMAIN_TASK_IDS:
      self._task_behavior = {
          key: ActorCritic(config, self.act_space, self.tfstep)
          for key in common.DOMAIN_TASK_IDS[domain]
      }
    else:
      self._task_behavior = ActorCritic(config, self.act_space, self.tfstep)
    if config.expl_behavior == 'greedy':
      self._expl_behavior = self._task_behavior
    else:
      self._expl_behavior = getattr(expl, config.expl_behavior)(
          self.config, self.act_space, self.wm, self.tfstep,
          lambda seq: self.wm.heads['reward'](seq['feat']).mode())

  @tf.function
  def policy(self, obs, state=None, mode='train', task=''):
    obs = tf.nest.map_structure(tf.tensor, obs)
    tf.py_function(lambda: self.tfstep.assign(
        int(self.step), read_value=False), [], [])
    if state is None:
      latent = self.wm.rssm.initial(len(obs['reward']))
      action = tf.zeros((len(obs['reward']),) + self.act_space.shape)
      state = latent, action
    latent, action = state
    embed = self.wm.encoder(self.wm.preprocess(obs))
    sample = (mode == 'train') or not self.config.eval_state_mean
    latent, _ = self.wm.rssm.obs_step(
        latent, action, embed, obs['is_first'], sample)
    feat = self.wm.rssm.get_feat(latent)
    if mode == 'eval':
      if task == '':
        actor = self._task_behavior.actor(feat)
      else:
        actor = self._task_behavior[task].actor(feat)
      action = actor.mode()
      noise = self.config.eval_noise
    elif mode == 'explore':
      actor = self._expl_behavior.actor(feat)
      action = actor.sample()
      noise = self.config.expl_noise
    elif mode == 'train':
      if task == '':
        actor = self._task_behavior.actor(feat)
      else:
        actor = self._task_behavior[task].actor(feat)
      action = actor.sample()
      noise = self.config.expl_noise
    action = common.action_noise(action, noise, self.act_space)
    outputs = {'action': action}
    state = (latent, action)
    return outputs, state
  
  @tf.function
  def train(self, data, state=None):
    metrics = {}
    state, outputs, mets = self.wm.train(data, state)
    metrics.update(mets)
    start = outputs['post']
    if isinstance(self._task_behavior, dict):
       for key in self._task_behavior.keys():
          reward = lambda seq: self.wm.heads['reward_' + key](seq['feat']).mode()
          mets, _ = self._task_behavior[key].train(
              self.wm, start, data['is_terminal'], reward)
          metrics.update(**{k+'_'+key: v for k, v in mets.items()})
    else:
      reward = lambda seq: self.wm.heads['reward'](seq['feat']).mode()
      task_met, _ = self._task_behavior.train(
          self.wm, start, data['is_terminal'], reward)
      metrics.update(task_met)
    if self.config.expl_behavior != 'greedy':
      _, mets, seq = self._expl_behavior.train(start, outputs, data)
      metrics.update({'expl_' + key: value for key, value in mets.items()})
      return state,