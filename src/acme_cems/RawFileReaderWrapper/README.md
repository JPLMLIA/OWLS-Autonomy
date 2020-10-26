# The RawFileReader .Net Implementation

This is a C# project wrapping the Thermo Fisher .Net implementation of RawFileReader.
It reads in a raw file and produces a JSON as an intermediate step in `src/cli/ACME_raw_data_converter.py`

This project should never need to be run independently, but if needed, usage (Mac/Linux) is:

`cd src/RawFileReaderWrapper/RawFileReaderWrapper/bin/Release/`

`mono RawFileReaderWrapper.exe [raw_file] [out_dir] [label]`

where

- `[raw_file]` is the full path to the raw file to be converted
- `[out_dir]` is the full path to the desired output directory of the resulting json
- `[label]` is the desired filename (without extension) of the resulting json

## Recompiling

Recompiling the .exe is done via msbuild from the directory containing the .sln solution file:

`cd src/RawFileReaderWrapper`

`msbuild /p:Configuration=Release`

## Dependencies

- mono
- dotnet (for msbuild)
