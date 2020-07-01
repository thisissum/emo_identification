class StopWordFilter(object):
    def __init__(self, stopword_path):
        self.stopword_path = stopword_path

        with open(stopword_path, 'r') as f:
            self.stopwords = set(f.readlines())

    def _filt(self, sentence):
        # sentence: List[str]
        output = filter(lambda x: x not in self.stopwords, sentence)
        return list(output)

    def step(self, inputs):
        """
        Filt stopwords from sentences after word-cutting
        Args:
            inputs: List[List[str]] or List[str], where str is cut.
        Returns:
            outputs: List[List[str]] or List[str], without stopwords.
        """
        if isinstance(inputs[0], list):
            outputs = []
            for sentence in inputs:
                outputs.append(self._filt(sentence))
        else:
            outputs = self._filt(inputs)
        return outputs