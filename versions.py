import glob
import random

def is_valid(version, seed=None):
    versions, seeds = get_versions_and_seeds()
    if seed is None:
        return version not in versions
    else:
        return (version, seed) not in zip(versions, seeds)


def get_versions_and_seeds():
    files = glob.glob('results/*/')
    versions = []
    seeds = []
    for f in files:
        f2 = f[len('results/'):-1] # go from 'results/*/' to '*'
        # go from 1278_0 to 1278, 0
        v, s = f2.split('_')
        versions.append(int(v))
        seeds.append(int(s))

    return versions, seeds


def next_version():
    versions, seeds = get_versions_and_seeds()
    available_versions = [i for i in range(100000) if i not in versions]
    if not available_versions:
        raise ValueError('No available versions. increase the max version number.')
    else:
        # this way, we can mostly ignore concurrent requests
        return random.choice(available_versions)

# this way, we can call it as a bash script, and it will "return" this value
print(next_version())
