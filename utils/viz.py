from tqdm import tqdm

class TqdmExtraFormat(tqdm):
    """Provides a `minutes per iteration` format parameter"""
    @property
    def format_dict(self):
        d = super(TqdmExtraFormat, self).format_dict
        rate_min = '{:.2f}'.format(1/d["rate"] / 60) if d["rate"] else '?'
        d.update(rate_min=(rate_min + ' min/' + d['unit']))
        return d