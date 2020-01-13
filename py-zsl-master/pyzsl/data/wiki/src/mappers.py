import logging
import ujson as json

from pyzsl.utils.general import obj_name, maybe_tqdm_open


class Mapper:
    """ Used to transform input file into output file(s), one line at a time.

    Use `.apply` method to perform the transformation
    Override `_in` and `_out` methods to add custom loading / dumping.

    Notes
    -----
    Handling mutliple output files
        See `.apply` method docstring for details.

    Overriding loading / saving:
        See JsonMapper for the concrete example
        Mapper reads a line and passes it through `_in` function, before applying transforms
        Similarly, it uses `_out` function on an item before storing it in file
    """

    _in  = lambda self, x: str(x)
    _out = lambda self, x: str(x)

    def __init__(self, input, *outputs):
        self.input   = input
        self.outputs = outputs

        self._logger = logging.getLogger(self.__class__.__name__)

    def _write(self, item, f):
        if item is None:
            return

        if not isinstance(item, list):
            item = [item]

        for entry in item:
            print(self._out(entry), file=f)

    def apply(self, *fns):
        """ Apply a chain of functions in `fns` to each line in input files, and write result(s) to output file(s)

        * The first function will receive the input line transformed with `self._in` method.
        * Then the output of previous function is directly fed to the next one in the chain.
        * If any of the functions returns `None`, we move to the next line
        * Last function should return a single item (if single output file was given) or a tuple of items.
        * If tuple is returned, it should contain one item per one output file specified
        * items which are None are ignored.
        """

        self._logger.info(f'Applying {len(fns)} function(s) '
                          f'mapping input {self.input} to '
                          f'outputs {self.outputs}.')

        f_outs = [open(o, 'w')
                  for o in self.outputs]

        names = {fn: obj_name(fn)
                 for fn in fns}
        mapping = {name: {'in': 0, 'out': 0} for name in names}

        try:
            with maybe_tqdm_open(self.input, flag=True) as f_in:
                for line in f_in:

                    line = self._in(line.strip())

                    for fn in fns:

                        if line is None:
                            break

                        line = fn(line)

                        mapping[fn]['in']  += 1
                        mapping[fn]['out'] += int(line is not None)

                    if not isinstance(line, tuple):
                        line = (line, )

                    for item, f_out in zip(line, f_outs):
                        self._write(item, f_out)

                for k, v in mapping.items():
                    self._logger.debug(f"Function {names[k]} received {v['in']} inputs and returned {v['out']} outputs. ")

        finally:
            for f in f_outs:
                f.close()


class JsonMapper(Mapper):
    """ Mapper which assumes that each input line is a valid `json` object,
    and that each output can be serialized as `json`.
    """

    _in  = json.loads
    _out = json.dumps
