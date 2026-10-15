#!/usr/bin/env python3
"""Calibrate CSV board readings and export unsmoothed plots and data.

Usage: python calibrate_trials.py [--input-dir PATH] [--output-dir PATH]
Requires numpy and matplotlib. CSV values are used directly as board readout,
confirmed by the user. Source CSV files are never modified.
"""
import argparse
import csv
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

FACTOR, OFFSET, EXPONENT = 0.00091759, 331.11493, 1.00730
COLORS = {'Control': '#0072B2', 'Mental': '#8C4AA0', 'Physical': '#D55E00'}

def load_trial(path):
    rows, issues = [], []
    with path.open(newline='', encoding='utf-8-sig') as f:
        for line, row in enumerate(csv.DictReader(f), 2):
            raw = row.get('EDA0/nS', '') or ''
            match = re.fullmatch(r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*', raw)
            try:
                timestamp = float(row['Timestamp'])
                if not match:
                    raise ValueError('No numeric reading')
                board = float(match.group(1))
                note = '; '.join(x for x in [row.get('Notes'), match.group(2)] if x)
                if not np.isfinite(timestamp) or not np.isfinite(board):
                    raise ValueError('Nonfinite value')
                calibrated = FACTOR * (board - OFFSET)**EXPONENT if board > OFFSET else np.nan
                if note or not np.isfinite(calibrated):
                    issues.append([path.name, line, timestamp, raw, note or 'Outside calibration domain'])
                rows.append([timestamp, board, calibrated, note])
            except (ValueError, TypeError) as err:
                issues.append([path.name, line, row.get('Timestamp'), raw, str(err)])
    if not rows:
        raise ValueError('No valid rows in ' + str(path))
    if any(rows[i][0] < rows[i-1][0] for i in range(1, len(rows))):
        raise ValueError('Nonmonotonic timestamps in ' + str(path))
    time = np.array([r[0] for r in rows])
    return rows, (time-time[0])/60, np.array([r[2] for r in rows]), issues

def style(ax, title):
    ax.set_title(title, loc='left', fontweight='bold', fontsize=11)
    ax.set_xlabel('Elapsed time (min)')
    ax.set_ylabel('Calibrated skin conductance (µS)')
    ax.grid(True, alpha=.22)
    ax.spines[['top', 'right']].set_visible(False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path.home()/'Desktop'/'Capstone Data Analysis')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'output')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'savefig.dpi':200})
    files = sorted(args.input_dir.glob('*.csv'))
    if not files:
        raise ValueError('No CSV trials found')
    trials, all_issues, summary = [], [], []
    for path in files:
        rows, time, values, issues = load_trial(path)
        category = next((c for c in COLORS if c in path.stem), 'Other')
        trials.append((path.stem, category, rows, time, values))
        all_issues.extend(issues)
        with (out/(path.stem+' calibrated.csv')).open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp_unix_s', 'Elapsed_time_min', 'Board_readout', 'Calibrated_conductance_uS', 'Notes'])
            writer.writerows([r[0], t, r[1], r[2], r[3]] for r,t in zip(rows,time))
        summary.append([path.stem, len(rows), time[-1], np.nanmin(values), np.nanmean(values), np.nanmax(values), len(issues)])
    with PdfPages(out/'all_calibrated_trials.pdf') as pdf:
        fig, axes = plt.subplots(4, 2, figsize=(14,13), constrained_layout=True)
        for ax, (name,category,rows,time,values) in zip(axes.flat, trials):
            ax.plot(time,values,color=COLORS.get(category,'gray'),lw=.8)
            style(ax,name)
            for row,t in zip(rows,time):
                if 'Connection dropped' in row[3]:
                    ax.axvline(t,color='gray',ls=':',lw=1,label='Connection-drop record')
            if any('Connection dropped' in r[3] for r in rows):
                ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(trials):]:
            ax.set_visible(False)
        fig.suptitle('Shivani — calibrated trial recordings\nUnsmoothed samples; each trial starts at elapsed time zero',fontsize=16)
        fig.savefig(out/'all_calibrated_trials.png')
        pdf.savefig(fig)
        plt.close(fig)
        for name,category,rows,time,values in trials:
            fig, ax = plt.subplots(figsize=(11,4.5),constrained_layout=True)
            ax.plot(time,values,color=COLORS.get(category,'gray'),lw=.9)
            style(ax,name)
            for row,t in zip(rows,time):
                if 'Connection dropped' in row[3]:
                    ax.axvline(t,color='gray',ls=':',label='Connection-drop record')
            if any('Connection dropped' in r[3] for r in rows): ax.legend()
            fig.savefig(out/(name+' calibrated.png'))
            pdf.savefig(fig)
            plt.close(fig)
        fig,axes = plt.subplots(3,1,figsize=(11,11),constrained_layout=True,sharey=True)
        for ax,category in zip(axes,COLORS):
            for name,cat,rows,time,values in trials:
                if cat == category: ax.plot(time,values,lw=.9,label=name.replace('Shivani ',''))
            style(ax, category+' trials')
            ax.legend(fontsize=9)
        fig.suptitle('Calibrated trials by condition — common conductance scale',fontsize=15)
        fig.savefig(out/'condition_comparison.png')
        pdf.savefig(fig)
        plt.close(fig)
    for name,header,data in [
        ('trial_summary.csv',['Trial','Samples','Duration_min','Minimum_uS','Mean_uS','Maximum_uS','Flagged_rows'],summary),
        ('data_quality.csv',['Source','CSV_line','Timestamp','Original_reading','Issue'],all_issues)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(header);writer.writerows(data)
    (out/'README.txt').write_text('Calibration from supplied screenshot:\nConductance (µS) = 0.00091759 × (Board readout − 331.11493)^1.00730\n\nCSV EDA0/nS values used directly as board readout, as confirmed by user.\nElapsed time uses actual timestamps, relative to first sample of each trial.\nNo smoothing, baseline subtraction, or resampling applied.\nNumeric readings with appended Connection dropped text are retained and flagged.\nValues at or below the calibration offset are exported as NaN.\nOverview panels use independent y-axis limits; condition comparison uses a common y-axis.\nCalibration uncertainty and validity range were not provided.\n')
    print('Created',len(trials),'calibrated trials in',out)
    print('Flagged rows:',len(all_issues))

if __name__ == '__main__':
    main()
