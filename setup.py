from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
  name='hash_encoding',
  ext_modules=[cpp_extension.CppExtension('hash_encoding', ['src/csrc/hash_encoding.cu'])],
  cmdclass={'build_ext': cpp_extension.BuildExtension}
)
