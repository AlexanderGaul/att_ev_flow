from torch.utils.data import Dataset


class CombinedProcessedSequence(Dataset) :
    def __init__(self, partseqs, processors, seq_idx=None) :
        self.partseqs = partseqs
        self.preps = processors
        self.seq_idx = seq_idx

    def get(self, idx) :
        return {k : v
                for sub in self.partseqs
                for k, v in sub[idx].items()}

    def process(self, data) :
        for func in self.preps :
            data = func(data)
        return data

    def get_item_id(self, idx) :
        if self.seq_idx is not None :
            return (self.seq_idx, idx)
        else :
            return (idx, )

    def __len__(self) :
        return len(self.partseqs[0])

    def __getitem__(self, idx) :
        return {**self.process(self.get(idx)),
                'frame_id' : self.get_item_id(idx)}