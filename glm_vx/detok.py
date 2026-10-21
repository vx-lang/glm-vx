"""Incremental detokenization for streamed text deltas.

A token can end in the middle of a UTF-8 character (byte-level BPE) or decode differently once
the next token arrives. Text is emitted only when the decoded window is stable: it grew, it still
starts with what was decoded before, and it does not end in U+FFFD (an incomplete byte sequence).
Held-back text is released by a later token or by flush(). This is the prefix/read offset window
used by TGI and vLLM, so each step decodes a short window instead of the whole output.
"""
import os


class IncrementalText:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids = []
        self.prefix = 0  # start of the decode window (already-emitted context)
        self.read = 0    # tokens before this index have had their text emitted

    def _window(self):
        before = self.tokenizer.decode(self.ids[self.prefix:self.read])
        return before, self.tokenizer.decode(self.ids[self.prefix:])

    def push(self, token):
        """Add one token; return the newly stable text ('' when held back or untokenized)."""
        self.ids.append(token)
        if self.tokenizer is None:
            return ''
        before, after = self._window()
        if len(after) > len(before) and after.startswith(before) and not after.endswith('�'):
            self.prefix, self.read = self.read, len(self.ids)
            return after[len(before):]
        return ''

    def flush(self):
        """Release whatever is still held back at the end of the output."""
        if self.tokenizer is None or self.read == len(self.ids):
            return ''
        before, after = self._window()
        self.prefix = self.read = len(self.ids)
        return after[len(os.path.commonprefix([before, after])):]
