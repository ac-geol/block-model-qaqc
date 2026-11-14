# Getting started (Windows)

This guide takes you from nothing to a finished QA/QC report, step by step. You don't need to
know Python. You will copy and paste a few commands once, during setup. After that, running the
checks is a drag and drop.

Using a Mac or Linux? See [Mac and Linux](#mac-and-linux) at the end.

**What you need**

- A Windows PC with an internet connection (only needed during setup).
- Your block model exported as a CSV file (from Datamine, Isatis or Leapfrog).
- About 15 minutes for the first setup.

---

## Step 1: Get the tool

1. Go to <https://github.com/ac-geol/block-model-qaqc>.
2. Click the green **Code** button, then **Download ZIP**.
3. Open your Downloads folder, right-click the ZIP file and choose **Extract All...**.
4. Move the extracted folder somewhere simple that is **not** synced by OneDrive, for example
   `C:\Tools\block-model-qaqc`. (Synced folders can lock files while the tool is using them.)

The folder now holds the tool. You will work with these files:

| File | What it is for |
|---|---|
| `setup_windows.bat` | One-time setup. |
| `run_demo.bat` | Try the tool on a made-up model. |
| `run_qaqc.bat` | Check your own model. |
| `run_notebook.bat` | Optional: run the checks step by step in a notebook. |
| `config\project_example.yaml` | The settings file you copy for each project. |

---

## Step 2: Install "uv" (once per PC)

`uv` is a small, free program that installs Python and the packages this tool needs. It does
not need administrator rights and does not change other programs.

1. Open **PowerShell**: press the Windows key, type `powershell` and press Enter. A window
   with a text prompt opens. This is the "terminal": you type a command and press Enter.
2. Copy this line, paste it into the PowerShell window (right-click pastes), and press Enter:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   If your company blocks this, try `winget install --id=astral-sh.uv -e` instead, or ask IT
   to install **uv** (from astral.sh) or **Python 3.11** for you.
3. **Close the PowerShell window** when it finishes. A new window is needed before Windows can
   find `uv`.

To check it worked: open PowerShell again, type `uv --version` and press Enter. You should see
a version number, such as `uv 0.9.5`.

---

## Step 3: Set up the tool (once)

Double-click **`setup_windows.bat`** in the tool folder.

A black window opens and shows its progress. It:

- makes a private copy of Python just for this tool, in a folder called `.venv`. This copy is
  called a "virtual environment". It keeps the tool's packages separate from anything else on
  your PC;
- downloads the packages the tool uses.

When it says **Setup finished**, press any key to close it.

> If Windows shows "Windows protected your PC", click **More info**, then **Run anyway**. This
> warning appears for any downloaded `.bat` file. Read the file in Notepad first if you want
> to see what it does: it only runs the two `uv` commands described above.

---

## Step 4: Try the demo

Double-click **`run_demo.bat`**.

It makes a small, made-up block model with known problems, runs every check on it, and opens
the report in your web browser. This shows you what to expect before you use real data. The
demo files are written to the `demo_project` folder.

---

## Step 5: Make a settings file for your project

Each project has its own settings file. It tells the tool where your block model is and what
your project's limits are.

1. Copy `config\project_example.yaml` into your project folder, for example
   `C:\Projects\MyMine\qaqc_settings.yaml`.
2. Open the copy in Notepad: right-click it, choose **Open with**, then **Notepad**.
3. Change these lines:
   - `project_name`: any name you like, in quotes.
   - `units`: `metric` or `imperial`.
   - `block_model_csv`: the full path to your block model CSV, written with **forward
     slashes**, for example `C:/Projects/MyMine/model_v3.csv`.
   - `output_dir`: the folder where the report should go, for example
     `C:/Projects/MyMine/qaqc_results/`.
   - `elements`: one line per element, with its unit, ceiling and cutoff.
   - `classification`: the class codes your model uses for Measured, Indicated, Inferred,
     Exploration Target (ET) and Unclassified.
   - `support`: your project's classification criteria.

   Every setting has a short note above it in the file.
4. Save the file.

**Rules for the settings file.** It is a YAML file. These rules prevent most mistakes:

- Lines that start with `#` are notes and are ignored.
- Indentation matters. Use **spaces, never tabs**, and keep each setting lined up with the
  ones around it.
- Keep the space after each colon: `units: metric`, not `units:metric`.
- Use forward slashes in paths. If you paste a path with backslashes, put it in **single**
  quotes: `'C:\Projects\MyMine\model.csv'`.

You don't have to get it right first time. If a setting is wrong or misspelled, the run stops
and tells you which line to fix.

---

## Step 6: Run the checks

**Drag your settings file onto `run_qaqc.bat`.**

A window opens and shows the progress. A large model (millions of blocks) can take a few
minutes on the first run. Later runs on the same file are faster, because a cleaned copy of
the model is saved.

When it finishes, the window lists:

- any warnings,
- how many blocks were flagged by each check,
- the files it wrote.

The report opens in your web browser.

If the window says **The run stopped**, read the message under it. It says what to fix,
usually a setting or a column name. Fix the settings file, save it, and drag it onto
`run_qaqc.bat` again.

---

## Step 7: Read the results

These files are written to the `output_dir` folder from your settings file:

| File | What it is |
|---|---|
| `BM_QAQC_Report.html` | The report. Double-click it to open it in a web browser. It works offline and can be emailed. |
| `flagged_blocks_leapfrog.csv` | Every flagged block, as points to import into Leapfrog. The report's *Flag register* tab gives a filter for each flag. |
| `audit_bundle.json` | Every result in one file, for records and for comparing model versions. |

Start with the **Overview** tab of the report. It has a status summary and the warnings. It
also shows which column of your file was used for what. Check that table first: if a column
was matched wrongly, every later result is affected.

---

## Common problems

| Message or problem | What to do |
|---|---|
| `'uv' is not recognized` | Close PowerShell and open a new window (step 2.3). If it still fails, repeat step 2. |
| `The tool is not set up yet` | Run `setup_windows.bat` first (step 3). |
| `... this setting is not recognised (check the spelling)` | A setting name in your settings file is misspelled. The message names it. |
| `... uses settings that have been renamed or removed` | Your settings file is from an older version. The message says what to change. |
| `More than one column could be ...` | The tool can't tell which column to use. Copy the line it suggests into your settings file. |
| `Cannot open ... close it and try again` | The file is open in Excel or a browser. Close it and run again. |
| `Rotated block models are not supported yet` | The model is rotated. Rotated models can't be checked yet. |
| Results look wrong for units | Check `units:` (metric or imperial) and each element's `unit:` in the settings file. |
| Need to run it again | Drag the settings file onto `run_qaqc.bat` again. Close the old report in Excel or the browser first. |

---

## Updating the tool

Download the ZIP again (step 1) and replace the old folder. Your settings files are in your
project folders, so they are not affected. Then double-click `setup_windows.bat` once more to
bring the packages up to date.

---

## Other ways to run it

These are optional.

**Step by step in a notebook.** `notebooks\block_model_qaqc.ipynb` runs the same checks one
step at a time, with a note on what to look for at each step. Double-click
**`run_notebook.bat`** and the notebook opens in your browser. In **Step 0**, change
`DEMO = True` to `DEMO = False` and set `CONFIG_PATH` to your settings file, in quotes, with
forward slashes. Then choose **Run → Run All Cells**. Close the black window when you are
finished.

**From PowerShell.** Open PowerShell in the tool folder: in File Explorer, open the folder,
click the address bar, type `powershell` and press Enter. Then:

```powershell
.venv\Scripts\python.exe -m qaqc run "C:/Projects/MyMine/qaqc_settings.yaml" --open
```

To check only how the columns of a file are matched (fast; it reads only the top of the file):

```powershell
.venv\Scripts\python.exe -m qaqc schema "C:/Projects/MyMine/model_v3.csv" --config "C:/Projects/MyMine/qaqc_settings.yaml"
```

---

## Mac and Linux

Open the Terminal app in the tool folder, then run these once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[notebook]"
```

To try the demo, and to run your own settings file:

```bash
.venv/bin/python -m qaqc demo --open
.venv/bin/python -m qaqc run path/to/qaqc_settings.yaml --open
```
