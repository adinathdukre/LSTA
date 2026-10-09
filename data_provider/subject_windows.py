class SubjectIndex:
    '''To return support and query windows for a given subject ID'''

    
    def __init__(self, pool):
        self.subject_to_class = {}
        self.support_pool = {}        # store support and query windows keeping subject Id as the key in a dictionary
        self.query_pool = {}

        for c in pool.classes:
            for s, idx in pool.subject_windows_by_class[c].items():
                self.subject_to_class[s] = c
                X = pool.X_by_class[c][idx]  # idx preserves temporal order (boolean-mask + np.where)
                half = X.shape[0] // 2
                self.support_pool[s] = X[:half]
                self.query_pool[s] = X[half:]
        self.subjects = sorted(self.subject_to_class.keys())


    def draw_episode(self, subject, rng, support_sizes, n_query):
        """draws a single set of support/query windows for a given subject
            input:
                subject: str, subject ID
                rng: np.random.Generator, random number generator
                support_sizes: list of int, sizes of support sets to draw
                n_query: int, number of query windows to draw
            output:
                nested: dict, mapping support size -> support windows (np.ndarray)
                query: np.ndarray, query windows
                class_label: int, class label of the subject
        """

        qp = self.query_pool[subject]
        n_q = min(n_query, qp.shape[0])
        q_idx = rng.choice(qp.shape[0], size=n_q, replace=False)    
        query = qp[q_idx]

        sp = self.support_pool[subject]
        perm = rng.permutation(sp.shape[0])
        nested = {}
        for k in support_sizes:
            kk = min(k, sp.shape[0])
            nested[k] = sp[perm[:kk]]
        return nested, query, self.subject_to_class[subject]
