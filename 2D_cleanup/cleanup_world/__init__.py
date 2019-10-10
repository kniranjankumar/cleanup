from gym.envs.registration import register

register(
    id='2DCleanup-v0',
    entry_point='cleanup_world.envs:CleanupWorld',
)