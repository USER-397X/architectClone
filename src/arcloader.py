from pathlib import Path
import json
import numpy as np
from tqdm import tqdm



class ArcDataset(object):
    def __init__(self, challenge, solutions={}, keys=None):
        self.keys = [k for k in keys]
        base_keys = set(map(self.get_base_key, self.keys))
        self.challenge = {k: challenge[k] for k in base_keys}
        self.solutions = {k: solutions[k] for k in base_keys if k in solutions}

    def __len__(self):
        """Allows you to call len() on a dataset object."""
        return len(self.keys)

    def __getitem__(self, index):
        """
        This is the core method that makes the object subscriptable.
        It fetches and returns the data for a single task.
        """
        if index >= len(self.keys):
            raise IndexError("Index out of range")

        # 1. Get the specific key for the requested index
        key = self.keys[index]

        # 2. Parse the key to find the base task ID
        base_key, reply_num = self.get_base_key_and_reply_num(key)

        # 3. Retrieve the raw data for that task
        task_challenge = self.challenge[base_key]
        task_solution = self.solutions[base_key][reply_num]

        # 4. Assemble and return a clean dictionary for inspection
        return {
            'key': key,
            'train_examples': task_challenge['train'],
            'test_input': task_challenge['test'][reply_num]['input'],
            'solution': task_solution
        }

    @classmethod
    def load_from_rearc(cls, path, n, sizes, seed, mix_datasets={}, shuffle=True):  # loader for ReArc
        np.random.seed(seed)
        keys = [[] for _ in range(n)]
        challenge = {}
        solutions = {}
        sizes = list(sizes)
        
        # Convert the input path string to a Path object
        path = Path(path)

        # Use pathlib to build the path and open the file
        metadata_file = path / 'metadata.json'
        with metadata_file.open() as f:
            metadata = json.load(f)

        for key in tqdm(sorted(metadata.keys()), desc="load dataset 're-arc'"):
            # Use pathlib for the second file path as well
            task_file = path / 'tasks' / f'{key}.json'
            with task_file.open() as f:
                tasks = np.random.permutation(json.load(f)).tolist()

            next_sizes = []
            for epoch in range(n):
                if not len(next_sizes):
                    next_sizes = np.random.permutation(sizes).tolist()
                next_size_with_test = 1 + next_sizes.pop()
                base_key = f'rearc-{key}{epoch:02x}'
                keys[epoch].append(f'{base_key}_0')
                challenge[base_key] = {'train': [], 'test': []}
                solutions[base_key] = []
                for _ in range(next_size_with_test):
                    if not len(tasks):
                        raise RuntimeError('Not enough examples - generate more re-arc examples or reduce epochs.')
                    challenge[base_key]['train'].append({k: v for k, v in tasks.pop().items()})
                challenge[base_key]['test'].append(challenge[base_key]['train'].pop())
                solutions[base_key].append(challenge[base_key]['test'][-1].pop('output'))
        
        if shuffle:
            keys = [np.random.permutation(epoch) for epoch in keys]
        keys = [k for epoch in keys for k in epoch]


        return cls(keys=keys, challenge=challenge, solutions=solutions)

    @staticmethod
    def get_base_key_and_reply_num(key):
        key_num = key.split('.', 1)[0]
        base_key, reply_num = key_num.split('_') if '_' in key_num else (key_num, -1)
        return base_key, int(reply_num)

    @classmethod
    def get_base_key(cls, key):
        return cls.get_base_key_and_reply_num(key)[0]

