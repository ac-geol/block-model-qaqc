# Header samples from real exports

Put real export samples here, one file per software and model type:

    datamine_regular.csv      datamine_sub_blocked.csv
    isatis_regular.csv        isatis_percentage.csv
    leapfrog_regular.csv      leapfrog_sub_blocked.csv

Each file should hold any lines above the column names, the column-name row and about five
data rows (values can be changed or scrambled for confidentiality). If possible include a
rotated model and the estimation detail columns (sample counts, pass, kriging variance, and so
on).

To see how a sample is read:

    .venv/bin/python -m qaqc schema tests/fixtures/headers/leapfrog_sub_blocked.csv

Once the result has been reviewed and is correct, add an entry for the file to
`expected.yaml` in this folder (same layout as `provisional/expected.yaml`). From then on the
test suite checks that the file keeps being read the same way. Samples with no entry are
skipped.

The `provisional/` folder holds stand-in samples made by the synthetic generator in each
software's *expected* style. They show the format, but they are not real exports.
