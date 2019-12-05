import gym
import numpy as np
import tensorflow as tf
from tensorflow import saved_model as sm
from easy_rl.utils.window_stat import WindowStat
from easy_rl.utils.gym_wrapper.atari_wrapper import make_atari, wrap_deepmind
import time


def main():
    gym_env = gym.make("CartPole-v0")

    atari_env = make_atari("PongNoFrameskip-v4")
    atari_env = wrap_deepmind(
        env=atari_env,
        frame_stack=True,
        clip_rewards=False,
        episode_life=True,
        wrap_frame=True,
        frame_resize=42)

    # replace the following env according to your saved_model
    # env = atari_env
    env = gym_env

    with tf.Session() as sess:
        path = 'dump_dir'
        MetaGraphDef = tf.saved_model.loader.load(
            sess, tags=[sm.tag_constants.SERVING], export_dir=path)

        # get SignatureDef protobuf
        SignatureDef_d = MetaGraphDef.signature_def
        SignatureDef = SignatureDef_d["predict_results"]

        # get inputs/outputs TensorInfo protobuf
        ph_inputs = {}
        for name, ts_info in SignatureDef.inputs.items():
            ph_inputs[name] = sm.utils.get_tensor_from_tensor_info(
                ts_info, sess.graph)

        outputs = {}
        for name, ts_info in SignatureDef.outputs.items():
            outputs[name] = sm.utils.get_tensor_from_tensor_info(
                ts_info, sess.graph)

        for name, ph in ph_inputs.items():
            print(name, ph)

        for name, ts in outputs.items():
            print(name, ts)

        len_window = WindowStat("length", 50)
        reward_window = WindowStat("reward", 50)
        for i in range(100):
            ob = env.reset()
            env.render()
            time.sleep(0.2)
            done = False
            episode_len = 0
            episode_reward = .0

            while not done:
                action = sess.run(
                    outputs["output_actions"],
                    feed_dict={
                        ph_inputs["obs_ph"]: [np.asarray(ob)],
                        ph_inputs["deterministic_ph"]: True
                    })
                next_ob, reward, done, info = env.step(action[0])
                env.render()
                time.sleep(0.1)
                episode_reward += reward
                episode_len += 1
                ob = next_ob

            len_window.push(episode_len)
            reward_window.push(episode_reward)
            print(reward_window)
            print(len_window)


if __name__ == '__main__':
    main()
