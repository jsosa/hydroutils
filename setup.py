from numpy.distutils.core import setup
from numpy.distutils.core import Extension

ext0 = Extension(name='hydroutils.lmoments.SAMLMU',
                 sources=['hydroutils/lmoments/SAMLMU.FOR'])
ext1 = Extension(name='hydroutils.lmoments.PELEXP',
                 sources=['hydroutils/lmoments/PELEXP.FOR'])
ext2 = Extension(name='hydroutils.lmoments.PELGAM',
                 sources=['hydroutils/lmoments/PELGAM.FOR'])
ext3 = Extension(name='hydroutils.lmoments.PELGEV',
                 sources=['hydroutils/lmoments/PELGEV.FOR'])
ext4 = Extension(name='hydroutils.lmoments.PELGLO',
                 sources=['hydroutils/lmoments/PELGLO.FOR'])
ext5 = Extension(name='hydroutils.lmoments.PELGNO',
                 sources=['hydroutils/lmoments/PELGNO.FOR'])
ext6 = Extension(name='hydroutils.lmoments.PELGPA',
                 sources=['hydroutils/lmoments/PELGPA.FOR'])
ext7 = Extension(name='hydroutils.lmoments.PELGUM',
                 sources=['hydroutils/lmoments/PELGUM.FOR'])
ext8 = Extension(name='hydroutils.lmoments.PELKAP',
                 sources=['hydroutils/lmoments/PELKAP.FOR'])
ext9 = Extension(name='hydroutils.lmoments.PELNOR',
                 sources=['hydroutils/lmoments/PELNOR.FOR'])
ext10 = Extension(name='hydroutils.lmoments.PELPE3',
                 sources=['hydroutils/lmoments/PELPE3.FOR'])
ext11 = Extension(name='hydroutils.lmoments.PELWAK',
                 sources=['hydroutils/lmoments/PELWAK.FOR'])
setup(
    name='hydroutils',
    version='0.1',
    description='Hydrological tools for daily problems',
    url='http://github.com/jsosa/hydroutils',
    author='Jeison Sosa',
    author_email='sosa.jeison@gmail.com',
    license='MIT',
    packages=['hydroutils'],
    zip_safe=False,
    ext_modules = [ext0,ext1,ext2,ext3,ext4,ext5,ext6,ext7,ext8,ext9,ext10,ext11]
    )
