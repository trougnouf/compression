"""Models used to seamlessly call standard compression methods with PyTorch."""
import os
import sys

sys.path.append("..")
from compression.models import abstract_model
from common.libs import stdcompression
from common.libs import pt_ops
from common.libs import pt_helpers
from common.libs import utilities
import torch

TMP_INIMG_DPATH = os.path.join("tmp", "inimg")
TMP_OUTIMG_DPATH = os.path.join("tmp", "outimg")


class Std_ImageCompressor(abstract_model.AbstractImageCompressor):
    """Abstract model used as base for other standards."""

    def __init__(self, quality, **kwargs):
        super().__init__(device=torch.device(type="cpu"))
        os.makedirs(TMP_INIMG_DPATH, exist_ok=True)
        os.makedirs(TMP_OUTIMG_DPATH, exist_ok=True)
        self.quality = quality

    def make_input_image_file(self, input_image) -> str:
        """Save tensor image to lossless file, return filepath."""
        if input_image.size(0) != 1:
            raise NotImplementedError()
        inimg_fn = str(pt_ops.fragile_checksum(input_image)) + ".png"
        tmp_lossless_img_fpath = os.path.join(TMP_INIMG_DPATH, inimg_fn)
        if not os.path.isfile(tmp_lossless_img_fpath):
            pt_helpers.tensor_to_imgfile(input_image, tmp_lossless_img_fpath)
        return tmp_lossless_img_fpath

    def forward(self, input_image):
        """
        Forward should:
            convert tensor to image file,
            encode an image file using the appropriate codec,
            get bitrate,
            decode if necessary and convert back to tensor.
        """
        in_fpath = self.make_input_image_file(input_image)
        out_fpath = os.path.join(
            TMP_OUTIMG_DPATH, type(self).__name__ + utilities.get_leaf(in_fpath)
        )
        if not self.REQ_DEC:
            out_fpath += "." + self.ENCEXT
        std_results = self.file_encdec(in_fpath, out_fpath, quality=self.quality)
        out_results = dict()
        out_results["bpp"] = (std_results["encsize"] * 8) / (
            input_image.shape[-2:].numel()
        )
        out_results["reconstructed_image"] = pt_helpers.fpath_to_tensor(
            out_fpath, batch=True
        )
        os.remove(out_fpath)
        return out_results


class JPEG_ImageCompressor(Std_ImageCompressor, stdcompression.JPG_Compression):
    """JPG PyTorch compression model interface."""

    def __init__(self, quality, **kwargs):
        super().__init__(quality=quality, **kwargs)


class BPG_ImageCompressor(Std_ImageCompressor, stdcompression.BPG_Compression):
    """BPG PyTorch compression model interface."""

    def __init__(self, quality, **kwargs):
        super().__init__(quality=quality, **kwargs)


class JPEGXS_ImageCompressor(Std_ImageCompressor, stdcompression.JPEGXS_Compression):
    """JPEG XS PyTorch compression model interface."""

    def __init__(self, quality, **kwargs):
        super().__init__(quality=quality, **kwargs)


class JPEGXL_ImageCompressor(Std_ImageCompressor, stdcompression.JPEGXL_Compression):
    """JPEG XL PyTorch compression model interface."""

    def __init__(self, quality, **kwargs):
        super().__init__(quality=quality, **kwargs)


classes_dict = {
    "bpg": BPG_ImageCompressor,
    "jpg": JPEG_ImageCompressor,
    "jxs": JPEGXS_ImageCompressor,
    "jxl": JPEGXL_ImageCompressor,
}
