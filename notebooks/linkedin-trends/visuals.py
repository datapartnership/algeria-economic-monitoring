"""
Visualization functions for LinkedIn migration analysis
"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_multi_country_radar(df, country_names, year=2019, column='industry_name', title='Industry Location Quotients', 
                            section_names=None, section_column='isic_section_name', figsize_per_plot=(6, 5), n_cols=2):
    """
    Radar chart showing multiple countries' LQ profiles across industries
    Can create subplots for multiple sections with a shared legend
    
    Args:
        df: DataFrame with LQ data
        country_names: List of country names to plot
        year: Year to plot
        column: Column name for categories (default: 'industry_name')
        title: Overall title for the plot
        section_names: Optional list of section names to create subplots for each section
        section_column: Column name for sections (default: 'isic_section_name')
        figsize_per_plot: Tuple of (width, height) for each subplot (default: (6, 5))
        n_cols: Number of columns in subplot grid (default: 2)
    """
    # Generate distinct colors for all countries
    if len(country_names) <= 10:
        colors = plt.cm.tab10.colors
    elif len(country_names) <= 20:
        colors = plt.cm.tab20.colors
    else:
        colors = plt.cm.hsv(np.linspace(0, 0.9, len(country_names)))
    
    # If section_names provided, create subplots
    if section_names:
        n_sections = len(section_names)
        cols = min(n_cols, n_sections)
        rows = (n_sections + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_plot[0]*cols, figsize_per_plot[1]*rows), 
                                subplot_kw=dict(projection='polar'))
        
        # Handle single subplot case
        if n_sections == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if n_sections > 1 else [axes]
        else:
            axes = axes.flatten()
        
        global_max = 0
        
        # Plot each section
        for idx, section_name in enumerate(section_names):
            ax = axes[idx]
            
            # Filter data for this section
            section_data = df[(df[section_column] == section_name) & (df['year'] == year)]
            
            if section_data.empty:
                ax.text(0.5, 0.5, f'No data for\n{section_name}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(section_name, size=11, fontweight='bold', pad=15)
                continue
            
            # Get categories for this section
            categories = sorted(section_data[column].unique().tolist())
            N = len(categories)
            
            if N == 0:
                continue
            
            # Compute angles
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            
            max_value = 0
            
            # Plot each country
            for i, country_name in enumerate(country_names):
                country_data = section_data[section_data['country_name'] == country_name]
                
                if country_data.empty:
                    continue
                
                # Prepare data
                country_dict = dict(zip(country_data[column], country_data['lq']))
                values = [country_dict.get(category, 0) for category in categories]
                values += values[:1]
                
                max_value = max(max_value, max(values[:-1]) if values[:-1] else 0)
                
                # Plot
                ax.plot(angles, values, 'o-', linewidth=1.5, 
                       color=colors[i], markersize=3, label=country_name if idx == 0 else "")
                ax.fill(angles, values, alpha=0.1, color=colors[i])
            
            # Add reference line
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.6, linewidth=1)
            
            # Customize subplot
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=7, rotation=45)
            ax.set_ylim(0, max(max_value * 1.15, 1.5))
            ax.set_title(section_name, size=11, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3)
            
            global_max = max(global_max, max_value)
        
        # Hide empty subplots
        for i in range(n_sections, len(axes)):
            axes[i].set_visible(False)
        
        # Add single legend
        handles, labels = axes[0].get_legend_handles_labels()
        # Add reference line to legend
        from matplotlib.lines import Line2D
        ref_line = Line2D([0], [0], color='red', linestyle='--', linewidth=1, alpha=0.6)
        handles.append(ref_line)
        labels.append('Income Group Avg (LQ=1)')
        
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.02), 
                  ncol=min(len(labels), 4), framealpha=0.9, fontsize=9)
        fig.suptitle(f'{title} ({year})', fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        plt.show()
        
    else:
        # Original single plot behavior
        all_data = df[df['year'] == year]
        categories = sorted(all_data[column].unique().tolist())
        
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(projection='polar'))
        
        max_value = 0
        
        # Plot each country
        for i, country_name in enumerate(country_names):
            country_data = df[(df['country_name'] == country_name) & (df['year'] == year)]
            
            if country_data.empty:
                print(f"Warning: No data found for {country_name} in {year}")
                continue
            
            country_dict = dict(zip(country_data[column], country_data['lq']))
            values = [country_dict.get(category, 0) for category in categories]
            values += values[:1]
            
            max_value = max(max_value, max(values[:-1]))
            
            ax.plot(angles, values, 'o-', linewidth=1.5, label=country_name, 
                    color=colors[i], markersize=4)
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9, rotation=45)
        
        # Add reference line
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.8, linewidth=1, 
                   label='Income Group Average (LQ=1)')
        
        ax.set_ylim(0, max_value * 1.15)
        ax.set_title(f'{title} ({year})', size=14, y=1.08, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), fontsize=9)
        
        plt.tight_layout()
        plt.show()


def plot_multi_country_radar_subplots(df, country_names, year=2019, column='isic_section_name', 
                                    cols=2):
    """
    Create individual radar charts for multiple countries in subplots
    
    Args:
        df: DataFrame with LQ data
        country_names: List of country names to plot
        year: Year to plot
        column: Column name for categories
        cols: Number of columns in subplot grid
    """
    # Calculate number of rows needed
    rows = (len(country_names) + cols - 1) // cols
    
    # Get categories from first country
    first_country_data = df[(df['country_name'] == country_names[0]) & (df['year'] == year)]
    categories = first_country_data[column].tolist()
    N = len(categories)
    
    # Compute angles
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 4*rows), 
                            subplot_kw=dict(projection='polar'))
    
    # Handle case where there's only one subplot
    if len(country_names) == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # Color for all countries (consistent)
    color = 'blue'
    
    # Find global max for consistent y-axis scaling
    global_max = 0
    for country_name in country_names:
        country_data = df[(df['country_name'] == country_name) & (df['year'] == year)]
        if not country_data.empty:
            global_max = max(global_max, country_data['lq'].max())
    
    # Plot each country
    for i, country_name in enumerate(country_names):
        ax = axes[i]
        
        country_data = df[(df['country_name'] == country_name) & (df['year'] == year)]
        
        if country_data.empty:
            ax.text(0.5, 0.5, f'No data\nfor {country_name}', 
                   transform=ax.transAxes, ha='center', va='center')
            continue
        
        # Prepare data
        country_data_sorted = country_data.set_index(column).reindex(categories)
        values = country_data_sorted['lq'].fillna(0).tolist()
        values += values[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color=color, markersize=2)
        ax.fill(angles, values, alpha=0.25, color=color)
        
        # Add reference line
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Customize each subplot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, global_max * 1.1)
        ax.set_title(country_name, size=12, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(country_names), len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'Industry Section Location Quotients by Country ({year})', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()


def plot_lq_heatmap(df, year=2023, income_group=None, column='isic_section_name'):
    """
    Heatmap showing LQ values across countries (rows) and ISIC sections (columns)
    Colors normalized by country (column-wise), so each country's gradient is independent
    """
    # Filter data
    if 'year' in df.columns:
        plot_data = df[df['year'] == year].copy()
    else:
        plot_data = df.copy()
    if income_group:
        plot_data = plot_data[plot_data['wb_income'] == income_group]
    
    # Pivot for heatmap
    heatmap_data = plot_data.pivot(
        index=column, 
        columns='country_name', 
        values='lq'
    )
    
    # Normalize each column (country) independently to 0-1 range
    # This makes the gradient relative to each country's min and max
    normalized_data = heatmap_data.copy()
    for col in normalized_data.columns:
        col_min = normalized_data[col].min()
        col_max = normalized_data[col].max()
        if col_max - col_min > 0:  # Avoid division by zero
            normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
        else:
            normalized_data[col] = 0.5  # If all values are the same, use middle value
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot with normalized data for colors, but show original values in annotations
    ax = sns.heatmap(
        normalized_data, 
        annot=heatmap_data,  # Show original values
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Normalized Location Quotient (by Country)'}
    )
    
    plt.title(f'Location Quotient by Country and {column} ({year})\n(Colors normalized by country)', 
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_comparative_lq(df, countries, section_name, year=2023):
    """
    Bar chart comparing LQ for specific sector across multiple countries
    """
    plot_data = df[
        (df['country_name'].isin(countries)) & 
        (df['isic_section_name'] == section_name) & 
        (df['year'] == year)
    ].copy()
    
    plot_data = plot_data.sort_values('lq', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if x > 1 else 'red' for x in plot_data['lq']]
    bars = ax.barh(plot_data['country_name'], plot_data['lq'], color=colors, alpha=0.7)
    
    # Add value labels
    for bar, value in zip(bars, plot_data['lq']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.2f}', ha='left', va='center')
    
    ax.axvline(x=1, color='black', linestyle='--', alpha=0.8, label='Income Group Average')
    ax.set_xlabel('Location Quotient')
    ax.set_title(f'{section_name} - Comparative Specialization ({year})')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_matplotlib_custom_tabs(df, tab_groups, x_col='year', y_col='column', group_col='c'):
    """
    Create Matplotlib subplots for different tab groups
    
    Args:
        df: DataFrame with columns 'year', 'c', 'column'
        tab_groups: Dictionary where keys are tab names and values are lists of 'c' values
                   Example: {'Tab 1': ['c1', 'c2'], 'Tab 2': ['c3', 'c4', 'c5']}
        x_col: Column name for x-axis (default: 'year')
        y_col: Column name for y-axis (default: 'column')
        group_col: Column name for grouping (default: 'c')
    """
    # Create a subplot for each tab group
    n_tabs = len(tab_groups)
    fig, axes = plt.subplots(n_tabs, 1, figsize=(12, 6 * n_tabs))
    
    # Handle single subplot case
    if n_tabs == 1:
        axes = [axes]
    
    # Color palette
    colors = plt.cm.tab10.colors
    
    # Create a plot for each tab group
    for idx, (tab_name, c_values) in enumerate(tab_groups.items()):
        ax = axes[idx]
        
        # Plot each 'c' value in this tab
        for i, c_value in enumerate(c_values):
            # Filter data for this 'c' value
            group_data = df[df[group_col] == c_value].sort_values(x_col)
            
            if group_data.empty:
                print(f"Warning: No data found for {group_col} = {c_value}")
                continue
            
            # Choose color
            color = colors[i % len(colors)]
            
            # Plot line with markers
            ax.plot(group_data[x_col], group_data[y_col], 
                   marker='o', linewidth=2.5, markersize=6,
                   color=color, alpha=0.8, label=str(c_value))
        
        # Customize the plot
        ax.set_title(f"{tab_name}: {', '.join(map(str, c_values))}", 
                    fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(x_col.title(), fontsize=12)
        ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Set integer ticks for year if applicable
        if x_col.lower() == 'year':
            ax.set_xticks(sorted(df[x_col].unique()))
    
    plt.tight_layout()
    plt.show()


def plot_source_country_tabs(df, source_countries, metric_col='s2d_d2s_ratio', top_dest_n=5):
    """
    Create Matplotlib stacked bar charts for specified source countries
    
    Args:
        df: DataFrame with columns 'year', 'source_country_region', 'dest_country_region', metric
        source_countries: List of source countries to create tabs for
        metric_col: Column to visualize (default: 's2d_d2s_ratio')
        top_dest_n: Number of top destination countries to include in stacks (default: 5)
    """
    n_countries = len(source_countries)
    fig, axes = plt.subplots(n_countries, 1, figsize=(12, 6 * n_countries))
    
    # Handle single subplot case
    if n_countries == 1:
        axes = [axes]
    
    plot_created = False
    
    for idx, source_country in enumerate(source_countries):
        ax = axes[idx]
        
        # Filter data for this source country
        source_data = df[df['source_country_region'] == source_country]
        
        if source_data.empty:
            print(f"Warning: No data found for source country: {source_country}")
            ax.text(0.5, 0.5, f'No data for {source_country}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(source_country, fontsize=14, fontweight='bold')
            continue
        
        # Find top N destination countries for this source
        top_destinations = (source_data.groupby('dest_country_region')[metric_col]
                           .sum()
                           .nlargest(top_dest_n)
                           .index.tolist())
        
        # Filter to top destinations
        filtered_data = source_data[source_data['dest_country_region'].isin(top_destinations)]
        
        if filtered_data.empty:
            ax.text(0.5, 0.5, f'No data for {source_country}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(source_country, fontsize=14, fontweight='bold')
            continue
        
        # Get years for this source country
        years = sorted(filtered_data['year'].unique())
        
        # Prepare data for stacked bars
        stacked_data = {}
        for dest in top_destinations:
            stacked_data[dest] = []
            for year in years:
                year_data = filtered_data[
                    (filtered_data['year'] == year) & 
                    (filtered_data['dest_country_region'] == dest)
                ][metric_col].sum()
                stacked_data[dest].append(year_data)
        
        # Create stacked bar chart
        bottom = np.zeros(len(years))
        colors = plt.cm.tab10.colors
        
        for i, (dest, values) in enumerate(stacked_data.items()):
            ax.bar(years, values, bottom=bottom, 
                  label=dest, color=colors[i % len(colors)], alpha=0.8)
            bottom += np.array(values)
        
        # Customize the plot
        ax.set_title(f"Migration Ratios from {source_country} (Top {top_dest_n} Destinations)",
                    fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(metric_col.replace('_', ' ').title(), fontsize=12)
        ax.legend(title='Destination Countries', loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        
        plot_created = True
    
    if plot_created:
        plt.tight_layout()
        plt.show()
    else:
        print("No data found for any of the specified source countries")
