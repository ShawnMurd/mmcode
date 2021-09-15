# Overview: mmcode

This project contains a Python script to plot mobile mesonet observations from [VORTEX2](http://www.vortex2.org/home/) on top of radar imagery (either NEXRAD or DOWs). Mobile mesonet data are represented using either color-coded station models or station models with text displaying thermodynamic quantities.


### Links

- [NEXRAD data](https://s3.amazonaws.com/noaa-nexrad-level2/index.html)
- [VORTEX2 mobile mesonet data orders](https://data.eol.ucar.edu/master_lists/generated/vortex2/)
- [VORTEX2 2009 Field Catalog](http://catalog.eol.ucar.edu/cgi-bin/vortex2_2009/report/index)
- [VORTEX2 2010 Field Catalog](http://catalog.eol.ucar.edu/cgi-bin/vortex2_2010/report/index)


### Contents

- **plot_mobile_mesonets.py**: Python script to plot mobile mesonet data on top of a radar field. Users should only have to edit the "Input Parameters" section of the program, which is near the beginning. More documentation is given within the script.
- **mm_plot_YYYYMMDD.py**: Specific examples using plot_mobile_mesonet.py. When changing the code, all four cases should be tested because each contains different nuances that might cause the code to fail.
- **README_NSSL-PSU_mobilemesonet**: Mobile mesonet data documentation from VORTEX2 principal investigators.
- **version_log**: Describes updates made to the original version of the coda (which was made ~2019). Has not been updated since the project moved to GitHub on 9/13/2021.
- **YYYYMMDD**: Folders containing the data necessary for the four test cases. Includes mobile mesonet data (files that begin with a "p"), radar data, and either a PNG or PDF image that should be created when running mm_plot_YYYYMMDD.py
