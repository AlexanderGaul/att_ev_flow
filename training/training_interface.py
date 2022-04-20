

class AbstractTrainer :
    def __init__(self):
        pass

    def batch_size(self, sample):
        return len(sample)

    # TODO: should we copy the sample create sample only with deviced elements
    def sample_to_device(self, sample, device):
        return sample.to(device)

    def forward(self, model, sample):
        return dict()

    def evaluate(self, sample, out):
        return dict()

    def statistics(self, sample, out, eval, return_separate=False):
        return dict()

    def visualize(self, visualize_frames, sample, out, eval):
        return dict()

    def visualize_item(self, idx, sample, out, eval,
                       write_label_on_img=False):
        return dict()