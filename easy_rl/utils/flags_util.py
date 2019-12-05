import tensorflow as tf

# cluster info
tf.flags.DEFINE_string("ps_hosts", "", "ps_hosts")
tf.flags.DEFINE_string("worker_hosts", "", "worker_hosts")
tf.flags.DEFINE_string("job_name", "", "job_name")
tf.flags.DEFINE_integer("task_index", -1, "task_index")

# ODPS used
tf.flags.DEFINE_string("tables", "", "tables names")
tf.flags.DEFINE_string("outputs", "", "output tables names")

# model save&export
tf.flags.DEFINE_string("checkpointDir", "",
                       "oss buckets for saving checkpoint")
tf.flags.DEFINE_string("buckets", "", "oss buckets")

# job partition
tf.flags.DEFINE_integer("num_actors", 0, "number of actors")
tf.flags.DEFINE_integer("num_memories", 0, "number of memories")

# configuration for agent&model
tf.flags.DEFINE_string("config", "",
                       "filename or the json string of the configuration")

FLAGS = tf.flags.FLAGS


def get_distributed_spec():
    if len(FLAGS.ps_hosts) == 0:
        distributed_spec = {}
    else:
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.worker_hosts.split(',')

        task_index = FLAGS.task_index
        job_name = FLAGS.job_name

        # create actor and memory from ps resource
        actor_hosts = [ps_hosts[i] for i in range(FLAGS.num_actors)]
        memory_hosts = [
            ps_hosts[i + FLAGS.num_actors] for i in range(FLAGS.num_memories)
        ]
        ps_hosts = ps_hosts[(FLAGS.num_actors + FLAGS.num_memories):]
        learner_hosts = worker_hosts

        if job_name == 'worker':
            job_name = 'learner'
        elif job_name == 'ps':
            if task_index < FLAGS.num_actors:
                job_name = 'actor'
            elif task_index < FLAGS.num_actors + FLAGS.num_memories:
                job_name = 'memory'
                task_index -= FLAGS.num_actors
            else:
                task_index -= (FLAGS.num_actors + FLAGS.num_memories)

        distributed_spec = dict(
            job_name=job_name,
            task_index=task_index,
            actor_hosts=actor_hosts,
            memory_hosts=memory_hosts,
            ps_hosts=ps_hosts,
            learner_hosts=learner_hosts)

    return distributed_spec
