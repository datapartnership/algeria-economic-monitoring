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


# ============================================================================
# LinkedIn Hiring Rate Visualization Functions
# ============================================================================

def clean_linkedin_data(excel_file, sheet_name="2A - LHR SA by Ctry", column_name='LHR (YOY)'):
    """
    Clean and prepare LinkedIn data from Excel file.
    
    Parameters:
    -----------
    excel_file : str
        Path to the Excel file
    sheet_name : str
        Name of the sheet to read
    column_name : str
        Name for the LHR column (default 'LHR (YOY)')
    
    Returns:
    --------
    pd.DataFrame
        Cleaned data with columns: Month, Country, and LHR column
    """
    try:
        data_corrected = pd.read_excel(excel_file, sheet_name=sheet_name, header=3)

        data_cleaned = data_corrected.rename(columns={
            'Unnamed: 1': 'Month',
            'Unnamed: 2': 'Country',
            'Unnamed: 3': column_name
        })

        data_cleaned = data_cleaned.dropna(subset=['Month', 'Country', column_name])
        data_cleaned = data_cleaned.iloc[1:]
        data_cleaned = data_cleaned.drop(columns=['Unnamed: 0'])

        # Clean up whitespace and standardize country names
        data_cleaned["Country"] = data_cleaned["Country"].str.strip()
        data_cleaned["Country"] = data_cleaned["Country"].replace({
            "Turkey": "Turkiye",
            "Türkiye": "Turkiye"
        })

        return data_cleaned

    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None


def create_country_plots(excel_file):
    """
    Create interactive plots showing LinkedIn Hiring Rate by Country with 2022+ filter toggle and download button.
    Automatically detects whether to use SA or non-SA data based on filename.
    
    Parameters:
    -----------
    excel_file : str
        Path to the Excel file containing LHR data
    """
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, HoverTool, Tabs, TabPanel, Toggle, CustomJS, Button
    from bokeh.layouts import column
    from bokeh.palettes import Category20
    from bokeh.io import output_file
    import os
    
    country_map = {
        'DZ': 'Algeria', 
        'MA': 'Morocco',
        'US': 'United States'
    }
    
    try:
        # Detect if filename contains "SA" to determine which sheet to use
        filename = os.path.basename(excel_file)
        if 'SA' in filename or '_SA_' in filename:
            sheet_name = "2A - LHR SA by Ctry"
        else:
            sheet_name = "2A - LHR by Ctry"
        
        df = clean_linkedin_data(excel_file, sheet_name=sheet_name)
        
        if df is None:
            print("Data cleaning failed. Please check your Excel file.")
            return
        
        # Compute global y-axis range for all countries
        global_min = df['LHR (YOY)'].min()
        global_max = df['LHR (YOY)'].max()

        tabs = []

        for idx, (country_code, country_name) in enumerate(country_map.items()):
            country_data = df[df['Country'] == country_name].copy()
            if country_data.empty:
                print(f"No data available for {country_name}.")
                continue

            # Keep datetime format for plotting
            country_data['Month'] = pd.to_datetime(country_data['Month'])
            country_data = country_data.sort_values('Month')

            # Add string version of Month for download
            country_data['Month_str'] = country_data['Month'].dt.strftime('%Y-%m')

            # Prepare sources
            full_data = country_data
            filtered_data = full_data[full_data['Month'] >= '2022-01-01']

            source = ColumnDataSource(full_data)
            source_filtered = ColumnDataSource(filtered_data)
            # Download source: string Month only
            source_download = ColumnDataSource(
                full_data[['Country', 'Month_str', 'LHR (YOY)']].rename(columns={'Month_str': 'Month'})
            )

            # Create plot
            p = figure(
                title=f"LinkedIn Hiring Rate in {country_name}",
                x_axis_type='datetime',
                width=800,
                height=500,
                background_fill_color="#f8f9fa",
                y_range=(global_min, global_max)
            )

            p.line(
                x='Month',
                y='LHR (YOY)',
                source=source,
                line_width=3,
                color=Category20[20][idx % 20]
            )

            hover = HoverTool(tooltips=[
                ("Month", "@Month{%b %Y}"),
                ("LHR (YOY)", "@{LHR (YOY)}{0.00}%")
            ], formatters={"@Month": "datetime"}, mode='vline')
            p.add_tools(hover)

            p.xaxis.axis_label = 'Month'
            p.yaxis.axis_label = 'LinkedIn Hiring Rate (YOY %)'
            p.xaxis.axis_label_text_font_size = '12pt'
            p.yaxis.axis_label_text_font_size = '12pt'
            p.xaxis.major_label_text_font_size = '10pt'
            p.yaxis.major_label_text_font_size = '10pt'
            p.title.text_font_size = '14pt'
            p.grid.grid_line_color = "gray"
            p.grid.grid_line_alpha = 0.3

            # Toggle button
            toggle = Toggle(label="Show only from 2022", button_type="success", active=False)

            callback = CustomJS(args=dict(
                toggle=toggle,
                source=source,
                full=source.data,
                filtered=source_filtered.data
            ), code="""
                source.data = toggle.active ? filtered : full;
                source.change.emit();
                toggle.label = toggle.active ? "Show full range" : "Show only from 2022";
                toggle.button_type = toggle.active ? "warning" : "success";
            """)
            toggle.js_on_change("active", callback)

            # Download button
            download_button = Button(label="Download CSV", button_type="primary")

            download_js = CustomJS(args=dict(source=source_download, name=country_name), code="""
                const data = source.data;
                const cols = ["Country", "Month", "LHR (YOY)"];
                const nrows = data[cols[0]].length;
                let csv = cols.join(",") + "\\n";
                for (let i = 0; i < nrows; i++) {
                    let row = cols.map(col => data[col][i]);
                    csv += row.join(",") + "\\n";
                }
                const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
                const link = document.createElement("a");
                link.href = URL.createObjectURL(blob);
                const filename = "linkedin_HR_" + name.replace(/ /g, "_") + ".csv";
                link.download = filename;
                link.click();
            """)
            download_button.js_on_click(download_js)

            # Combine all widgets
            layout = column(toggle, download_button, p)
            tab = TabPanel(child=layout, title=country_name)
            tabs.append(tab)

        if tabs:
            tabs_obj = Tabs(tabs=tabs)
            output_file("linkedin_hiring_rate_by_country.html")
            show(tabs_obj)
        else:
            print("No data available for any of the specified countries.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your Excel file structure and column names.")


def compare_sa_vs_nonsa_plots(sa_file, nonsa_file):
    """
    Create interactive plots comparing Seasonally Adjusted (SA) vs Non-SA LinkedIn Hiring Rates by Country.
    Shows both LHR lines on the same plot for each country with toggle and download features.
    
    Parameters:
    -----------
    sa_file : str
        Path to the Excel file containing seasonally adjusted LHR data
    nonsa_file : str
        Path to the Excel file containing non-seasonally adjusted LHR data
    """
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, HoverTool, Tabs, TabPanel, Toggle, CustomJS, Button
    from bokeh.layouts import column
    from bokeh.io import output_file
    
    country_map = {
        'DZ': 'Algeria', 
        'MA': 'Morocco',
        'US': 'United States'
    }
    
    try:
        # Load SA data
        sa_data = pd.read_excel(sa_file, sheet_name="2A - LHR SA by Ctry", header=3)
        
        sa_cleaned = sa_data.rename(columns={
            'Unnamed: 1': 'Month',
            'Unnamed: 2': 'Country',
            'Unnamed: 3': 'LHR'
        })
        sa_cleaned = sa_cleaned.dropna(subset=['Month', 'Country', 'LHR'])
        sa_cleaned = sa_cleaned.iloc[1:]
        sa_cleaned = sa_cleaned.drop(columns=['Unnamed: 0'])
        sa_cleaned["Country"] = sa_cleaned["Country"].str.strip()
        
        # Load Non-SA data
        nonsa_data = pd.read_excel(nonsa_file, sheet_name="2A - LHR by Ctry", header=3)
        
        nonsa_cleaned = nonsa_data.rename(columns={
            'Unnamed: 1': 'Month',
            'Unnamed: 2': 'Country',
            'Unnamed: 3': 'LHR'
        })
        nonsa_cleaned = nonsa_cleaned.dropna(subset=['Month', 'Country', 'LHR'])
        nonsa_cleaned = nonsa_cleaned.iloc[1:]
        nonsa_cleaned = nonsa_cleaned.drop(columns=['Unnamed: 0'])
        nonsa_cleaned["Country"] = nonsa_cleaned["Country"].str.strip()
        
        # Compute global y-axis range for both datasets
        global_min = min(sa_cleaned['LHR'].min(), nonsa_cleaned['LHR'].min())
        global_max = max(sa_cleaned['LHR'].max(), nonsa_cleaned['LHR'].max())

        tabs = []

        for idx, (country_code, country_name) in enumerate(country_map.items()):
            # Filter data for this country
            sa_country = sa_cleaned[sa_cleaned['Country'] == country_name].copy()
            nonsa_country = nonsa_cleaned[nonsa_cleaned['Country'] == country_name].copy()
            
            if sa_country.empty and nonsa_country.empty:
                print(f"No data available for {country_name}.")
                continue

            # Process SA data
            if not sa_country.empty:
                sa_country['Month'] = pd.to_datetime(sa_country['Month'])
                sa_country = sa_country.sort_values('Month')
                sa_country['Month_str'] = sa_country['Month'].dt.strftime('%Y-%m')
            
            # Process Non-SA data
            if not nonsa_country.empty:
                nonsa_country['Month'] = pd.to_datetime(nonsa_country['Month'])
                nonsa_country = nonsa_country.sort_values('Month')
                nonsa_country['Month_str'] = nonsa_country['Month'].dt.strftime('%Y-%m')

            # Prepare full data sources
            source_sa = ColumnDataSource(sa_country)
            source_nonsa = ColumnDataSource(nonsa_country)
            
            # Prepare filtered data (2022+)
            sa_filtered = sa_country[sa_country['Month'] >= '2022-01-01']
            nonsa_filtered = nonsa_country[nonsa_country['Month'] >= '2022-01-01']
            source_sa_filtered = ColumnDataSource(sa_filtered)
            source_nonsa_filtered = ColumnDataSource(nonsa_filtered)
            
            # Prepare download sources (combine both datasets)
            download_data = pd.concat([
                sa_country[['Country', 'Month_str', 'LHR']].rename(columns={'Month_str': 'Month', 'LHR': 'LHR_SA'}),
                nonsa_country[['Month_str', 'LHR']].rename(columns={'Month_str': 'Month', 'LHR': 'LHR_NonSA'})
            ], axis=1)
            download_data = download_data.loc[:, ~download_data.columns.duplicated()]
            source_download = ColumnDataSource(download_data)

            # Create plot
            p = figure(
                title=f"LinkedIn Hiring Rate Comparison: {country_name} (SA vs Non-SA)",
                x_axis_type='datetime',
                width=900,
                height=500,
                background_fill_color="#f8f9fa",
                y_range=(global_min, global_max)
            )

            # Plot SA line
            sa_line = p.line(
                x='Month',
                y='LHR',
                source=source_sa,
                line_width=3,
                color='#1f77b4',
                legend_label='LHR (Seasonally Adjusted)',
                alpha=0.8
            )

            # Plot Non-SA line
            nonsa_line = p.line(
                x='Month',
                y='LHR',
                source=source_nonsa,
                line_width=3,
                color='#ff7f0e',
                legend_label='LHR (Non-Seasonally Adjusted)',
                alpha=0.8
            )

            # Add hover tools
            hover_sa = HoverTool(
                renderers=[sa_line],
                tooltips=[
                    ("Month", "@Month{%b %Y}"),
                    ("LHR (SA)", "@LHR{0.00}%")
                ],
                formatters={"@Month": "datetime"},
                mode='vline'
            )
            hover_nonsa = HoverTool(
                renderers=[nonsa_line],
                tooltips=[
                    ("Month", "@Month{%b %Y}"),
                    ("LHR (Non-SA)", "@LHR{0.00}%")
                ],
                formatters={"@Month": "datetime"},
                mode='vline'
            )
            p.add_tools(hover_sa, hover_nonsa)

            # Styling
            p.xaxis.axis_label = 'Month'
            p.yaxis.axis_label = 'LinkedIn Hiring Rate (%)'
            p.xaxis.axis_label_text_font_size = '12pt'
            p.yaxis.axis_label_text_font_size = '12pt'
            p.xaxis.major_label_text_font_size = '10pt'
            p.yaxis.major_label_text_font_size = '10pt'
            p.title.text_font_size = '14pt'
            p.grid.grid_line_color = "gray"
            p.grid.grid_line_alpha = 0.3
            p.legend.location = "top_left"
            p.legend.click_policy = "hide"

            # Toggle button for 2022+ filter
            toggle = Toggle(label="Show only from 2022", button_type="success", active=False)

            callback = CustomJS(args=dict(
                toggle=toggle,
                source_sa=source_sa,
                source_nonsa=source_nonsa,
                full_sa=source_sa.data,
                full_nonsa=source_nonsa.data,
                filtered_sa=source_sa_filtered.data,
                filtered_nonsa=source_nonsa_filtered.data
            ), code="""
                if (toggle.active) {
                    source_sa.data = filtered_sa;
                    source_nonsa.data = filtered_nonsa;
                    toggle.label = "Show full range";
                    toggle.button_type = "warning";
                } else {
                    source_sa.data = full_sa;
                    source_nonsa.data = full_nonsa;
                    toggle.label = "Show only from 2022";
                    toggle.button_type = "success";
                }
                source_sa.change.emit();
                source_nonsa.change.emit();
            """)
            toggle.js_on_change("active", callback)

            # Download button
            download_button = Button(label="Download CSV", button_type="primary")

            download_js = CustomJS(args=dict(source=source_download, name=country_name), code="""
                const data = source.data;
                const cols = Object.keys(data);
                const nrows = data[cols[0]].length;
                let csv = cols.join(",") + "\\n";
                for (let i = 0; i < nrows; i++) {
                    let row = cols.map(col => data[col][i]);
                    csv += row.join(",") + "\\n";
                }
                const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
                const link = document.createElement("a");
                link.href = URL.createObjectURL(blob);
                const filename = "linkedin_HR_comparison_" + name.replace(/ /g, "_") + ".csv";
                link.download = filename;
                link.click();
            """)
            download_button.js_on_click(download_js)

            # Combine all widgets
            layout = column(toggle, download_button, p)
            tab = TabPanel(child=layout, title=country_name)
            tabs.append(tab)

        if tabs:
            tabs_obj = Tabs(tabs=tabs)
            output_file("linkedin_hiring_rate_sa_vs_nonsa_comparison.html")
            show(tabs_obj)
        else:
            print("No data available for any of the specified countries.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Please check your Excel file structure and column names.")


def plot_epr_growth_tabs(employment_pivot):
    """
    Create Bokeh tabs (one per country) plotting EPR_Growth_pct vs Year.
    Expects a DataFrame with columns: 'Country Name', 'Year', 'EPR_Growth_pct'.
    Includes an additional README tab describing data source and computation.
    
    Parameters:
    -----------
    employment_pivot : pd.DataFrame
        DataFrame with columns: Country Name, Year, EPR_Growth_pct
    """
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, HoverTool, Tabs, TabPanel, Div
    from bokeh.layouts import column
    from bokeh.io import output_file

    tabs = []

    # Ensure numeric year
    employment_pivot = employment_pivot.copy()
    employment_pivot['Year'] = pd.to_numeric(employment_pivot['Year'], errors='coerce')

    # Get list of countries dynamically
    countries = employment_pivot['Country Name'].dropna().unique()

    for country in countries:
        dfc = employment_pivot[employment_pivot['Country Name'] == country].dropna(subset=['EPR_Growth_pct'])
        if dfc.empty:
            continue

        source = ColumnDataSource(dfc)

        p = figure(
            title=f"{country} — Employment-to-Population Growth (YoY %)",
            width=850,
            height=400,
            background_fill_color="#f8f9fa"
        )

        p.line('Year', 'EPR_Growth_pct', source=source, line_width=3, color="#4a6fa5")
        p.scatter('Year', 'EPR_Growth_pct', source=source, size=7, color="#2ca02c", line_color="white", line_width=1.5)

        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Year", "@Year"),
            ("EPR Growth (YoY %)", "@EPR_Growth_pct{0.00}"),
        ])
        p.add_tools(hover)

        # Add axis labels and grid
        p.xaxis.axis_label = "Year"
        p.yaxis.axis_label = "EPR Growth (YoY %)"
        p.grid.grid_line_color = "gray"
        p.grid.grid_line_alpha = 0.3
        p.title.text_font_size = "14pt"

        tabs.append(TabPanel(child=column(p), title=country))

    # Add README tab
    readme_html = """
    <h3>About this visualization</h3>
    <p>This figure presents the <b>year-to-year growth</b> in the
    <i>Employment-to-Population Ratio</i> for each selected country.</p>

    <h4>Computation</h4>
    <p>
    The indicator plotted here (<b>EPR_Growth_pct</b>) is calculated as the
    year-over-year percentage change in the employment-to-population ratio:
    </p>
    <pre>
    EPR_Growth_pct = ((EPR<sub>t</sub> - EPR<sub>t-1</sub>) / EPR<sub>t-1</sub>) × 100
    </pre>

    <h4>Interpretation</h4>
    <p>
    Positive values indicate an increase in the share of employed people relative to the working-age population compared to the previous year.
    Negative values indicate a decline.
    </p>
    
    <h4>Data Source</h4>
    <p>
    World Bank – <a href="https://data.worldbank.org/indicator/SL.EMP.TOTL.SP.ZS?locations=DZ" target="_blank">
    Employment to population ratio, 15+, total (%) (modeled ILO estimate)</a>.
    </p>
    """

    readme_tab = TabPanel(child=Div(text=readme_html, width=850), title="README")
    tabs.append(readme_tab)

    # Display all tabs
    if tabs:
        tabs_obj = Tabs(tabs=tabs)
        output_file("employment_to_population_growth.html")
        show(tabs_obj)
    else:
        print("No data available for the selected countries.")


def clean_linkedin_data_industry(excel_file):
    """Clean and prepare LinkedIn industry data from Excel file."""
    try:
        # Try to load the primary sheet first
        try:
            data_corrected = pd.read_excel(excel_file, sheet_name="2B - LHR by Ctry, Ind", header=3)
        except Exception:
            # If primary sheet doesn't exist, try the alternative sheet
            data_corrected = pd.read_excel(excel_file, sheet_name="2B - LHR SA by Ctry, Ind", header=3)
        
        # Rename columns and clean data
        data_cleaned = data_corrected.rename(columns={
            'Unnamed: 1': 'Month',
            'Unnamed: 2': 'Country',
            'Unnamed: 3': 'Industry',
            'Unnamed: 4': 'LHR (YOY)'
        })
        
        # Remove rows with missing data
        data_cleaned = data_cleaned.dropna(subset=['Month', 'Country', 'Industry', 'LHR (YOY)'])
        
        # Remove the first row (duplicate headers)
        data_cleaned = data_cleaned.iloc[1:]
        
        # Remove the column containing only NaN values
        data_cleaned = data_cleaned.drop(columns=['Unnamed: 0'])
        
        return data_cleaned
        
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None


def create_industry_performance_plots(excel_file, country='Algeria'):
    """
    Create interactive plots showing industry performance by country, with toggle to filter from 2022 and CSV download.
    
    Parameters:
    -----------
    excel_file : str
        Path to the Excel file containing industry data
    country : str
        Country name to filter data for (default: 'Algeria')
    """
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, HoverTool, Tabs, TabPanel, Legend, Toggle, CustomJS, Button
    from bokeh.layouts import column
    from bokeh.palettes import Category20
    from bokeh.io import output_file
    import os

    try:
        # Clean the data
        df = clean_linkedin_data_industry(excel_file)
        if df is None:
            print("Data cleaning failed. Please check your Excel file.")
            return

        # Filter data for specified country
        bd_data = df[df['Country'] == country].copy()

        # Convert Month to datetime
        bd_data['Month'] = pd.to_datetime(bd_data['Month'])

        # Calculate average LHR for each industry
        industry_avg = bd_data.groupby('Industry')['LHR (YOY)'].mean().sort_values(ascending=False)

        # Split industries into quartiles (5 industries each)
        top_5 = industry_avg.head(5).index.tolist()
        upper_mid_5 = industry_avg.iloc[5:10].index.tolist()
        lower_mid_5 = industry_avg.iloc[10:15].index.tolist()
        bottom_5 = industry_avg.iloc[15:].index.tolist()

        # Define quartile categories
        categories = [
            ('Leading Industries', top_5, 'Top Performing Industries (1st Quartile)'),
            ('Growing Industries', upper_mid_5, 'Strong Performing Industries (2nd Quartile)'),
            ('Transitioning Industries', lower_mid_5, 'Moderate Performing Industries (3rd Quartile)'),
            ('Emerging Industries', bottom_5, 'Developing Industries (4th Quartile)')
        ]

        tabs = []

        # Create one tab per industry category
        for tab_title, industries, plot_title in categories:
            # Prepare full and filtered data
            plot_data_full = bd_data[bd_data['Industry'].isin(industries)].copy()
            plot_data_filtered = plot_data_full[plot_data_full['Month'] >= '2022-01-01'].copy()
            
            # Prepare download data with Month as string
            download_data = plot_data_full.copy()
            download_data['Month_str'] = download_data['Month'].dt.strftime('%Y-%m')
            
            # Create download source
            source_download = ColumnDataSource(download_data)

            # Create figure
            p = figure(
                title=f"LinkedIn Hiring Rate: {plot_title} in {country}",
                x_axis_type='datetime',
                width=800,
                height=500,
                background_fill_color="#f8f9fa"
            )

            legend_items = []
            js_sources = []

            for i, industry in enumerate(industries):
                color = Category20[10][i]

                full_industry_data = plot_data_full[plot_data_full['Industry'] == industry]
                filtered_industry_data = plot_data_filtered[plot_data_filtered['Industry'] == industry]

                source = ColumnDataSource(full_industry_data)
                source_full = ColumnDataSource(full_industry_data)
                source_filtered = ColumnDataSource(filtered_industry_data)

                line = p.line(
                    x='Month',
                    y='LHR (YOY)',
                    source=source,
                    line_width=3,
                    color=color
                )

                legend_items.append((f"{industry} (Avg: {industry_avg[industry]:.2f}%)", [line]))
                js_sources.append({'source': source, 'full': source_full, 'filtered': source_filtered})

            # Add hover
            p.add_tools(HoverTool(
                tooltips=[
                    ("Month", "@Month{%b %Y}"),
                    ("LHR (YOY)", "@{LHR (YOY)}{0.00}%"),
                    ("Industry", "@Industry")
                ],
                formatters={"@Month": "datetime"},
                mode='vline'
            ))

            # Axes labels and styling
            p.xaxis.axis_label = 'Month'
            p.yaxis.axis_label = 'LinkedIn Hiring Rate (YOY %)'
            p.xaxis.axis_label_text_font_size = '12pt'
            p.yaxis.axis_label_text_font_size = '12pt'
            p.xaxis.major_label_text_font_size = '10pt'
            p.yaxis.major_label_text_font_size = '10pt'
            p.title.text_font_size = '14pt'
            p.grid.grid_line_color = "gray"
            p.grid.grid_line_alpha = 0.3

            # Add legend
            legend = Legend(
                items=legend_items,
                orientation='horizontal',
                spacing=20,
                padding=10,
                click_policy="hide",
                label_text_font_size='9pt',
                border_line_color="gray",
                border_line_alpha=0.5,
                background_fill_alpha=0.7,
                nrows=3
            )
            p.add_layout(legend, 'below')

            # Toggle button
            toggle = Toggle(label="Filter from 2022", button_type="success", active=False)

            # JS callback for toggle
            callback_code = """
                for (let i = 0; i < sources.length; i++) {
                    sources[i].data = toggle.active ? filtered[i].data : full[i].data;
                    sources[i].change.emit();
                }
                toggle.label = toggle.active ? 'Show full range' : 'Filter from 2022';
                toggle.button_type = toggle.active ? 'warning' : 'success';
        """

            callback = CustomJS(args={
                "toggle": toggle,
                "sources": [src['source'] for src in js_sources],
                "full": [src['full'] for src in js_sources],
                "filtered": [src['filtered'] for src in js_sources]
                }, code=callback_code)

            toggle.js_on_change("active", callback)
            
            # Download button
            download_button = Button(label="Download CSV", button_type="primary")
            
            download_js = CustomJS(args=dict(source=source_download, category=tab_title), code="""
                const data = source.data;
                const cols = ["Month_str", "Industry", "LHR (YOY)"];
                const nrows = data['Month_str'].length;
                let csv = "Month,Industry,LHR_YOY\\n";
                for (let i = 0; i < nrows; i++) {
                    let row = [data['Month_str'][i], data['Industry'][i], data['LHR (YOY)'][i]];
                    csv += row.join(",") + "\\n";
                }
                const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
                const link = document.createElement("a");
                link.href = URL.createObjectURL(blob);
                const filename = "linkedin_LHR_industries_" + category.replace(/ /g, "_") + ".csv";
                link.download = filename;
                link.click();
            """)
            download_button.js_on_click(download_js)

            # Combine toggle, download, and plot
            tab_layout = column(toggle, download_button, p)
            tab = TabPanel(child=tab_layout, title=tab_title)
            tabs.append(tab)

        # Show tabs
        if tabs:
            tabs_layout = Tabs(tabs=tabs)
            output_file(f"linkedin_hiring_rate_industries_{country.replace(' ', '_')}.html")
            show(tabs_layout)
        else:
            print("No data available for visualization.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your Excel file structure and column names.")


def compare_industry_sa_vs_nonsa_plots(sa_file, nonsa_file, country='Algeria'):
    """
    Create interactive plots comparing Seasonally Adjusted (SA) vs Non-SA LinkedIn Hiring Rates by Industry.
    Shows both LHR lines on the same plot for each industry with toggle and download features.
    
    Parameters:
    -----------
    sa_file : str
        Path to the Excel file containing seasonally adjusted LHR industry data
    nonsa_file : str
        Path to the Excel file containing non-seasonally adjusted LHR industry data
    country : str
        Country name to filter data for (default: 'Algeria')
    """
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, HoverTool, Tabs, TabPanel, Toggle, CustomJS, Button
    from bokeh.layouts import column
    from bokeh.io import output_file
    
    try:
        # Load SA data
        sa_data = pd.read_excel(sa_file, sheet_name="2B - LHR SA by Ctry, Ind", header=3)
        
        sa_cleaned = sa_data.rename(columns={
            'Unnamed: 1': 'Month',
            'Unnamed: 2': 'Country',
            'Unnamed: 3': 'Industry',
            'Unnamed: 4': 'LHR (YOY)'
        })
        sa_cleaned = sa_cleaned.dropna(subset=['Month', 'Country', 'Industry', 'LHR (YOY)'])
        sa_cleaned = sa_cleaned.iloc[1:]
        sa_cleaned = sa_cleaned.drop(columns=['Unnamed: 0'])
        sa_cleaned["Country"] = sa_cleaned["Country"].str.strip()
        sa_cleaned["Industry"] = sa_cleaned["Industry"].str.strip()
        
        # Load Non-SA data
        nonsa_data = pd.read_excel(nonsa_file, sheet_name="2B - LHR by Ctry, Ind", header=3)
        
        nonsa_cleaned = nonsa_data.rename(columns={
            'Unnamed: 1': 'Month',
            'Unnamed: 2': 'Country',
            'Unnamed: 3': 'Industry',
            'Unnamed: 4': 'LHR (YOY)'
        })
        nonsa_cleaned = nonsa_cleaned.dropna(subset=['Month', 'Country', 'Industry', 'LHR (YOY)'])
        nonsa_cleaned = nonsa_cleaned.iloc[1:]
        nonsa_cleaned = nonsa_cleaned.drop(columns=['Unnamed: 0'])
        nonsa_cleaned["Country"] = nonsa_cleaned["Country"].str.strip()
        nonsa_cleaned["Industry"] = nonsa_cleaned["Industry"].str.strip()
        
        # Filter for specified country
        sa_country = sa_cleaned[sa_cleaned['Country'] == country].copy()
        nonsa_country = nonsa_cleaned[nonsa_cleaned['Country'] == country].copy()
        
        if sa_country.empty and nonsa_country.empty:
            print(f"No data available for {country}.")
            return
        
        # Get list of industries (from both datasets)
        industries = sorted(set(sa_country['Industry'].unique().tolist() + nonsa_country['Industry'].unique().tolist()))
        
        # Compute global y-axis range for both datasets
        global_min = min(sa_country['LHR (YOY)'].min(), nonsa_country['LHR (YOY)'].min())
        global_max = max(sa_country['LHR (YOY)'].max(), nonsa_country['LHR (YOY)'].max())

        tabs = []

        for idx, industry in enumerate(industries):
            # Filter data for this industry
            sa_industry = sa_country[sa_country['Industry'] == industry].copy()
            nonsa_industry = nonsa_country[nonsa_country['Industry'] == industry].copy()
            
            # Skip if both are empty OR if only one dataset exists (we only want industries in both)
            if sa_industry.empty or nonsa_industry.empty:
                continue

            # Process SA data - need to reset index after filtering
            sa_industry = sa_industry.copy()
            sa_industry['Month'] = pd.to_datetime(sa_industry['Month'])
            sa_industry = sa_industry.sort_values('Month').reset_index(drop=True)
            
            # Process Non-SA data - need to reset index after filtering
            nonsa_industry = nonsa_industry.copy()
            nonsa_industry['Month'] = pd.to_datetime(nonsa_industry['Month'])
            nonsa_industry = nonsa_industry.sort_values('Month').reset_index(drop=True)

            # Prepare full data sources
            source_sa = ColumnDataSource(sa_industry)
            source_nonsa = ColumnDataSource(nonsa_industry)
            
            # Prepare filtered data (2022+)
            sa_filtered = sa_industry[sa_industry['Month'] >= '2022-01-01'].copy()
            nonsa_filtered = nonsa_industry[nonsa_industry['Month'] >= '2022-01-01'].copy()
            source_sa_filtered = ColumnDataSource(sa_filtered)
            source_nonsa_filtered = ColumnDataSource(nonsa_filtered)
            
            # Prepare download sources (combine both datasets)
            sa_download = sa_industry[['Country', 'Industry', 'Month', 'LHR (YOY)']].copy()
            sa_download['Month'] = sa_download['Month'].dt.strftime('%Y-%m')
            sa_download = sa_download.rename(columns={'LHR (YOY)': 'LHR_SA'})
            
            nonsa_download = nonsa_industry[['Month', 'LHR (YOY)']].copy()
            nonsa_download['Month'] = nonsa_download['Month'].dt.strftime('%Y-%m')
            nonsa_download = nonsa_download.rename(columns={'LHR (YOY)': 'LHR_NonSA'})
            
            download_data = pd.concat([sa_download, nonsa_download], axis=1)
            download_data = download_data.loc[:, ~download_data.columns.duplicated()]
            source_download = ColumnDataSource(download_data)

            # Create plot
            p = figure(
                title=f"LinkedIn Hiring Rate: {industry} in {country} (SA vs Non-SA)",
                x_axis_type='datetime',
                width=900,
                height=500,
                background_fill_color="#f8f9fa",
                y_range=(global_min, global_max)
            )

            # Plot SA line
            sa_line = p.line(
                x='Month',
                y='LHR (YOY)',
                source=source_sa,
                line_width=3,
                color='#1f77b4',
                legend_label='LHR (Seasonally Adjusted)',
                alpha=0.8
            )

            # Plot Non-SA line
            nonsa_line = p.line(
                x='Month',
                y='LHR (YOY)',
                source=source_nonsa,
                line_width=3,
                color='#ff7f0e',
                legend_label='LHR (Non-Seasonally Adjusted)',
                alpha=0.8
            )

            # Add hover tools
            hover_sa = HoverTool(
                renderers=[sa_line],
                tooltips=[
                    ("Month", "@Month{%b %Y}"),
                    ("LHR (SA)", "@{LHR (YOY)}{0.00}%"),
                    ("Industry", "@Industry")
                ],
                formatters={"@Month": "datetime"},
                mode='vline'
            )
            hover_nonsa = HoverTool(
                renderers=[nonsa_line],
                tooltips=[
                    ("Month", "@Month{%b %Y}"),
                    ("LHR (Non-SA)", "@{LHR (YOY)}{0.00}%"),
                    ("Industry", "@Industry")
                ],
                formatters={"@Month": "datetime"},
                mode='vline'
            )
            p.add_tools(hover_sa, hover_nonsa)

            # Styling
            p.xaxis.axis_label = 'Month'
            p.yaxis.axis_label = 'LinkedIn Hiring Rate (%)'
            p.xaxis.axis_label_text_font_size = '12pt'
            p.yaxis.axis_label_text_font_size = '12pt'
            p.xaxis.major_label_text_font_size = '10pt'
            p.yaxis.major_label_text_font_size = '10pt'
            p.title.text_font_size = '14pt'
            p.grid.grid_line_color = "gray"
            p.grid.grid_line_alpha = 0.3
            p.legend.location = "top_left"
            p.legend.click_policy = "hide"

            # Toggle button for 2022+ filter
            toggle = Toggle(label="Show only from 2022", button_type="success", active=False)

            callback = CustomJS(args=dict(
                toggle=toggle,
                source_sa=source_sa,
                source_nonsa=source_nonsa,
                full_sa=source_sa.data,
                full_nonsa=source_nonsa.data,
                filtered_sa=source_sa_filtered.data,
                filtered_nonsa=source_nonsa_filtered.data
            ), code="""
                if (toggle.active) {
                    source_sa.data = filtered_sa;
                    source_nonsa.data = filtered_nonsa;
                    toggle.label = "Show full range";
                    toggle.button_type = "warning";
                } else {
                    source_sa.data = full_sa;
                    source_nonsa.data = full_nonsa;
                    toggle.label = "Show only from 2022";
                    toggle.button_type = "success";
                }
                source_sa.change.emit();
                source_nonsa.change.emit();
            """)
            toggle.js_on_change("active", callback)

            # Download button
            download_button = Button(label="Download CSV", button_type="primary")

            download_js = CustomJS(args=dict(source=source_download, industry=industry, country=country), code="""
                const data = source.data;
                const cols = Object.keys(data);
                const nrows = data[cols[0]].length;
                let csv = cols.join(",") + "\\n";
                for (let i = 0; i < nrows; i++) {
                    let row = cols.map(col => data[col][i]);
                    csv += row.join(",") + "\\n";
                }
                const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
                const link = document.createElement("a");
                link.href = URL.createObjectURL(blob);
                const filename = "linkedin_HR_industry_" + industry.replace(/ /g, "_") + "_" + country.replace(/ /g, "_") + ".csv";
                link.download = filename;
                link.click();
            """)
            download_button.js_on_click(download_js)

            # Combine all widgets
            layout = column(toggle, download_button, p)
            tab = TabPanel(child=layout, title=industry)
            tabs.append(tab)

        if tabs:
            tabs_obj = Tabs(tabs=tabs)
            output_file(f"linkedin_hiring_rate_industry_sa_vs_nonsa_{country.replace(' ', '_')}.html")
            show(tabs_obj)
        else:
            print("No data available for any of the specified industries.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Please check your Excel file structure and column names.")


def plot_relative_penetration_by_peer_groups(excel_file):
    """
    Plot relative skill group penetration by peer groups with tabbed interface and CSV download.
    
    Parameters:
    -----------
    excel_file : str
        Path to the Excel file containing skill penetration data
    """
    from bokeh.plotting import figure, show
    from bokeh.io import output_file
    from bokeh.models import (
        ColumnDataSource, HoverTool, Legend, LegendItem,
        Tabs, TabPanel, Button, CustomJS
    )
    from bokeh.palettes import Category10
    from bokeh.transform import dodge
    from bokeh.layouts import column
    from utils import clean_skill_penetration_data
    
    df = clean_skill_penetration_data(excel_file)
    if df is None:
        return

    # Peer groups
    peer_groups = {
        "MENA (regional peers)": [
            "Algeria", "Egypt", "Iraq", "Jordan", "Morocco", "Tunisia"
        ],
        "Non-MENA (structural peers)": [
            "Algeria", "Uzbekistan", "Ecuador", "Peru", "Ghana", "Vietnam", "Colombia"
        ],
        "Non-MENA (aspirational peers)": [
            "Algeria", "Chile", "Kazakhstan", "Malaysia", "Poland", "Romania", "Turkey"
        ]
    }

    # Preferred skill order if available
    preferred_skill_order = [
        "Soft Skills", "Tech Skills", "Business Skills",
        "Disruptive Tech", "Green Skills"
    ]

    tabs = []

    for group_name, country_list in peer_groups.items():
        df_group = df[df['Country'].isin(country_list)].copy()
        if df_group.empty:
            print(f"No data for group: {group_name}")
            continue

        # Use only relative penetration
        grp = (
            df_group
            .groupby(['Skill', 'Country'])['Relative']
            .mean()
            .reset_index()
        )

        # Order skills
        skills_in_data = grp['Skill'].unique().tolist()
        # keep preferred order when available
        skill_order = [s for s in preferred_skill_order if s in skills_in_data]
        for s in skills_in_data:
            if s not in skill_order:
                skill_order.append(s)

        # Pivot: rows = Skill, cols = Country
        pivot = grp.pivot(index='Skill', columns='Country',
                          values='Relative') \
                    .reindex(skill_order)

        countries_in_data = [c for c in country_list if c in pivot.columns]
        if not countries_in_data:
            print(f"No matching countries with data in group: {group_name}")
            continue

        pivot = pivot[countries_in_data].reset_index()
        pivot.columns.name = None  # clean up
        source = ColumnDataSource(pivot)
        
        # Prepare download data
        download_data = pivot.copy()
        source_download = ColumnDataSource(download_data)

        # Figure
        p = figure(
            x_range=skill_order,
            height=450,
            width=900,
            title=f"Relative Skill Group Penetration – {group_name}",
            toolbar_location="above",
            background_fill_color="#ffffff"
        )

        n_countries = len(countries_in_data)
        # total width of group; each bar = group_width / n_countries
        group_width = 0.8
        bar_width = group_width / n_countries
        start = -group_width / 2 + bar_width / 2

        colors = Category10[max(3, min(10, n_countries))]

        legend_items = []

        for i, country in enumerate(countries_in_data):
            offset = start + i * bar_width

            r = p.vbar(
                x=dodge('Skill', offset, range=p.x_range),
                top=country,
                width=bar_width * 0.9,
                source=source,
                color=colors[i % len(colors)],
                name=country
            )

            # Hover only for this renderer
            hover = HoverTool(
                renderers=[r],
                tooltips=[
                    ("Country", country),
                    ("Skill group", "@Skill"),
                    ("Relative penetration", f"@{country}{{0.00}}")
                ]
            )
            p.add_tools(hover)

            legend_items.append(LegendItem(label=country, renderers=[r]))

        # Combine legend
        legend = Legend(items=legend_items)
        p.add_layout(legend, 'below')
        p.legend.orientation = "horizontal"
        p.legend.location = "center"
        p.legend.label_text_font_size = "9pt"

        p.xaxis.axis_label = "Skill group"
        p.yaxis.axis_label = "Relative skill group penetration"
        p.xaxis.major_label_orientation = 0.8
        p.y_range.start = 0

        # Remove vertical grid lines for cleaner look
        p.xgrid.grid_line_color = None
        
        # Download button
        download_button = Button(label="Download CSV", button_type="primary")
        
        # Build column list for CSV download
        csv_cols = ["Skill"] + countries_in_data
        
        download_js = CustomJS(args=dict(source=source_download, group_name=group_name, cols=csv_cols), code="""
            const data = source.data;
            const nrows = data['Skill'].length;
            let csv = cols.join(",") + "\\n";
            for (let i = 0; i < nrows; i++) {
                let row = cols.map(col => data[col][i]);
                csv += row.join(",") + "\\n";
            }
            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            const filename = "skill_penetration_" + group_name.replace(/ /g, "_").replace(/[()]/g, "") + ".csv";
            link.download = filename;
            link.click();
        """)
        download_button.js_on_click(download_js)

        # Combine plot and download button
        layout = column(download_button, p)
        tab = TabPanel(child=layout, title=group_name)
        tabs.append(tab)

    if tabs:
        tabs_obj = Tabs(tabs=tabs)
        output_file("skill_penetration_by_peer_groups.html")
        show(tabs_obj)
    else:
        print("No valid data to plot.")


def plot_relative_penetration_mena_only(excel_file):
    """
    Plot relative skill penetration for MENA countries by industry with dropdown and CSV download.
    
    Parameters:
    -----------
    excel_file : str
        Path to the Excel file containing skill penetration by industry data
    """
    from bokeh.plotting import figure, show
    from bokeh.io import output_file
    from bokeh.models import (
        ColumnDataSource, HoverTool, Legend, LegendItem,
        CustomJS, Dropdown, Button
    )
    from bokeh.layouts import column
    from bokeh.palettes import Category10
    from bokeh.transform import dodge
    from utils import clean_skill_penetration_by_industry_data
    
    df = clean_skill_penetration_by_industry_data(excel_file)
    if df is None:
        return

    # ---- MENA regional peers only ----
    mena_countries = ["Algeria", "Egypt", "Iraq", "Jordan", "Morocco", "Tunisia"]
    df_mena = df[df["Country"].isin(mena_countries)].copy()
    if df_mena.empty:
        print("No data for MENA peers.")
        return

    # Preferred skill order
    preferred_skill_order = [
        "Soft Skills", "Tech Skills", "Business Skills",
        "Disruptive Tech", "Green Skills"
    ]

    # Aggregate: mean Relative by (Industry, Skill, Country)
    g = (
        df_mena
        .groupby(["Industry", "Skill", "Country"])["Relative"]
        .mean()
        .reset_index()
    )

    # Countries actually present in data
    countries = [c for c in mena_countries if c in g["Country"].unique()]
    if not countries:
        print("No MENA countries with Relative data.")
        return

    # All industries in this subset
    industries = sorted(g["Industry"].unique().tolist())

    # -----------------------------
    # Build a dict: industry -> data dict for ColumnDataSource
    # data[industry] = {"Skill": [...], "Algeria":[...], "Egypt":[...], ...}
    # -----------------------------
    industry_data = {}

    for ind in industries:
        sub = g[g["Industry"] == ind].copy()

        # All skills in this industry
        skills_raw = sub["Skill"].unique().tolist()

        # enforce preferred skill order, then add remaining
        skill_order = [s for s in preferred_skill_order if s in skills_raw]
        for s in skills_raw:
            if s not in skill_order:
                skill_order.append(s)

        # Initialize dict for this industry
        data_dict = {"Skill": skill_order}

        for c in countries:
            vals = []
            for s in skill_order:
                match = sub[(sub["Country"] == c) & (sub["Skill"] == s)]
                if not match.empty:
                    vals.append(float(match["Relative"].iloc[0]))
                else:
                    vals.append(0.0)
            data_dict[c] = vals

        industry_data[ind] = data_dict

    # -----------------------------
    # Initial industry and source
    # -----------------------------
    initial_industry = industries[0]
    initial_data = industry_data[initial_industry]

    source = ColumnDataSource(initial_data)
    
    # Prepare download source (will be updated via JS)
    source_download = ColumnDataSource(initial_data)

    # -----------------------------
    # Figure
    # -----------------------------
    p = figure(
        x_range=initial_data["Skill"],
        height=450,
        width=900,
        title=f"Relative Skill Penetration – MENA Peers – {initial_industry}",
        toolbar_location="above",
        background_fill_color="#ffffff"
    )

    n = len(countries)
    group_width = 0.8
    bar_width = group_width / n
    start = -group_width / 2 + bar_width / 2

    colors = Category10[max(3, min(10, n))]
    legend_items = []

    for i, country in enumerate(countries):
        offset = start + i * bar_width

        r = p.vbar(
            x=dodge("Skill", offset, range=p.x_range),
            top=country,
            width=bar_width * 0.9,
            source=source,
            color=colors[i % len(colors)],
            name=country
        )

        hover = HoverTool(
            renderers=[r],
            tooltips=[
                ("Skill group", "@Skill"),
                ("Country", country),
                ("Relative penetration", f"@{country}{{0.00}}")
            ]
        )
        p.add_tools(hover)

        legend_items.append(LegendItem(label=country, renderers=[r]))

    legend = Legend(items=legend_items)
    p.add_layout(legend, "below")
    p.legend.orientation = "horizontal"
    p.legend.location = "center"
    p.legend.label_text_font_size = "9pt"

    p.xaxis.axis_label = "Skill group"
    p.yaxis.axis_label = "Relative skill group penetration"
    p.xaxis.major_label_orientation = 0.8
    p.y_range.start = 0
    p.xgrid.grid_line_color = None

    # -----------------------------
    # Dropdown to change industry
    # -----------------------------
    menu = [(ind, ind) for ind in industries]
    dropdown = Dropdown(
        label="Select industry",
        menu=menu,
        width=300
    )

    callback = CustomJS(
        args=dict(
            source=source,
            source_download=source_download,
            all_data=industry_data,
            plot=p
        ),
        code="""
            const selected = cb_obj.item;   // industry name from menu
            const new_data = all_data[selected];

            // Overwrite entire data of the source
            source.data = new_data;
            source.change.emit();
            
            // Update download source as well
            source_download.data = new_data;
            source_download.change.emit();

            // Update x-axis factors (skill groups)
            plot.x_range.factors = new_data['Skill'];

            // Update title
            plot.title.text = "Relative Skill Penetration – MENA Peers – " + selected;

            // Update dropdown label so user sees current selection
            cb_obj.label = "Select industry: " + selected;
        """
    )

    dropdown.js_on_event("menu_item_click", callback)
    
    # -----------------------------
    # Download button
    # -----------------------------
    download_button = Button(label="Download CSV", button_type="primary")
    
    csv_cols = ["Skill"] + countries
    
    download_js = CustomJS(args=dict(source=source_download, cols=csv_cols), code="""
        const data = source.data;
        const nrows = data['Skill'].length;
        let csv = cols.join(",") + "\\n";
        for (let i = 0; i < nrows; i++) {
            let row = cols.map(col => data[col][i]);
            csv += row.join(",") + "\\n";
        }
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = "skill_penetration_MENA_by_industry.csv";
        link.click();
    """)
    download_button.js_on_click(download_js)

    # -----------------------------
    # Show layout
    # -----------------------------
    layout = column(dropdown, download_button, p)
    output_file("skill_penetration_mena_by_industry.html")
    show(layout)


def plot_gender_skill_profile(excel_file, country='Algeria'):
    """
    Plot gender skill profile by industry showing relative importance with dropdown and CSV download.
    
    Parameters:
    -----------
    excel_file : str
        Path to the Excel file containing skill data by gender
    country : str
        Country name to filter data for (default: 'Algeria')
    """
    import pandas as pd
    from bokeh.io import output_file, show
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, CustomJS, Dropdown, HoverTool, Button
    from bokeh.layouts import column
    from bokeh.palettes import Category10
    
    # Load + prepare data
    df = pd.read_excel(excel_file, sheet_name="3B - SPP Ctry Ind Gen", header=5)
    df.columns = df.columns.str.strip()
    
    df = df[["Country", "Industry", "Skill", "Gender",
             "Relative Importance", "Skill Group Penetration"]]
    
    # Keep only specified country
    dz = df[df["Country"] == country].copy()
    
    if dz.empty:
        print(f"No data available for {country}")
        return
    
    # Pivot so we have female and male columns
    pivot = dz.pivot_table(
        index=["Industry", "Skill"],
        columns="Gender",
        values="Relative Importance",
        aggfunc="mean"
    ).reset_index()
    
    pivot = pivot.fillna(0)
    
    industries = sorted(pivot["Industry"].unique().tolist())
    
    # Build a dictionary {industry: dataframe_dict}
    industry_data = {}
    for ind in industries:
        sub = pivot[pivot["Industry"] == ind].copy()
        
        # Ensure columns exist
        if "female" not in sub.columns:
            sub["female"] = 0
        if "male" not in sub.columns:
            sub["male"] = 0
        
        industry_data[ind] = {
            "Skill": sub["Skill"].tolist(),
            "female": sub["female"].tolist(),
            "male": sub["male"].tolist()
        }
    
    # Initial industry to display
    initial = industries[0]
    
    source = ColumnDataSource(data=industry_data[initial])
    source_download = ColumnDataSource(data=industry_data[initial])
    
    # Create figure
    p = figure(
        x_range=(0, max(source.data["female"] + source.data["male"]) * 1.2),
        y_range=source.data["Skill"],
        width=750,
        height=450,
        title=f"Gender Skill Profile – {country} – {initial}",
        toolbar_location="above",
        background_fill_color="#ffffff"
    )
    
    # Lines connecting female → male
    p.segment(
        x0="female", y0="Skill",
        x1="male",   y1="Skill",
        source=source,
        line_width=2,
        color="gray"
    )
    
    # Female dots
    female_dots = p.scatter(
        x="female", y="Skill",
        size=9,
        color=Category10[3][0],
        source=source,
        legend_label="Female"
    )
    
    # Male dots
    male_dots = p.scatter(
        x="male", y="Skill",
        size=9,
        color=Category10[3][1],
        source=source,
        legend_label="Male"
    )
    
    # Tooltips
    p.add_tools(HoverTool(
        renderers=[female_dots],
        tooltips=[
            ("Skill", "@Skill"),
            ("Gender", "Female"),
            ("Relative Importance", "@female{0.000}")
        ]
    ))
    
    p.add_tools(HoverTool(
        renderers=[male_dots],
        tooltips=[
            ("Skill", "@Skill"),
            ("Gender", "Male"),
            ("Relative Importance", "@male{0.000}")
        ]
    ))
    
    # Cosmetics
    p.legend.location = "bottom_center"
    p.legend.orientation = "horizontal"
    p.xaxis.axis_label = "Relative Importance (TF-IDF weighted)"
    p.yaxis.axis_label = "Skill"
    p.xgrid.grid_line_color = None
    
    # Dropdown to change industry
    menu = [(ind, ind) for ind in industries]
    dropdown = Dropdown(
        label="Select Industry",
        menu=menu,
        width=300
    )
    
    # JS callback for dropdown
    callback = CustomJS(
        args=dict(
            source=source,
            source_download=source_download,
            all_data=industry_data,
            plot=p,
            country=country
        ),
        code="""
            const selected = cb_obj.item;
            const new_data = all_data[selected];
            
            source.data['Skill']  = new_data['Skill'];
            source.data['female'] = new_data['female'];
            source.data['male']   = new_data['male'];
            source.change.emit();
            
            // Update download source
            source_download.data['Skill']  = new_data['Skill'];
            source_download.data['female'] = new_data['female'];
            source_download.data['male']   = new_data['male'];
            source_download.change.emit();
            
            plot.y_range.factors = new_data['Skill'];
            plot.title.text = "Gender Skill Profile – " + country + " – " + selected;
            
            // Update dropdown label
            cb_obj.label = "Select Industry: " + selected;
        """
    )
    
    dropdown.js_on_event("menu_item_click", callback)
    
    # Download button
    download_button = Button(label="Download CSV", button_type="primary")
    
    download_js = CustomJS(args=dict(source=source_download), code="""
        const data = source.data;
        const cols = ["Skill", "female", "male"];
        const nrows = data['Skill'].length;
        let csv = "Skill,Female_Relative_Importance,Male_Relative_Importance\\n";
        for (let i = 0; i < nrows; i++) {
            let row = [data['Skill'][i], data['female'][i], data['male'][i]];
            csv += row.join(",") + "\\n";
        }
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = "gender_skill_profile.csv";
        link.click();
    """)
    download_button.js_on_click(download_js)
    
    # Show layout
    layout = column(dropdown, download_button, p)
    output_file(f"gender_skill_profile_{country.replace(' ', '_')}.html")
    show(layout)


def plot_skill_rank_flow_top10(skill_flow_df, country='Algeria', top_k=10):
    """
    Plot skill rank evolution over time by industry (top 10 skills per industry).
    
    Parameters:
    -----------
    skill_flow_df : pd.DataFrame
        DataFrame with columns: Country, Industry, Skill, Year, Skill Rank
    country : str
        Country name to filter data for (default: 'Algeria')
    top_k : int
        Number of top skills to display per industry (default: 10)
    """
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Filter to country
    df = skill_flow_df[skill_flow_df["Country"] == country].copy()
    
    if df.empty:
        print(f"No data available for {country}")
        return
    
    df = df.sort_values(["Industry", "Year", "Skill Rank"])
    industries = sorted(df["Industry"].unique())
    
    # Define a fixed palette of distinct colors
    palette10 = px.colors.qualitative.T10
    
    # Initial Industry
    initial_industry = industries[0]
    
    fig = go.Figure()
    trace_industry = []
    
    for ind in industries:
        df_ind = df[df["Industry"] == ind].copy()
        
        # Take the top k most frequent skills in this industry
        top_skills = df_ind["Skill"].value_counts().head(top_k).index.tolist()
        df_ind = df_ind[df_ind["Skill"].isin(top_skills)].copy()
        
        df_ind["Skill Rank"] = df_ind["Skill Rank"].astype(int)
        
        # Assign colors ONLY to these top skills
        color_map = {skill: palette10[i % len(palette10)] for i, skill in enumerate(top_skills)}
        
        for skill in top_skills:
            sd = df_ind[df_ind["Skill"] == skill].sort_values("Year")
            if sd.empty:
                continue
            
            visible_flag = (ind == initial_industry)
            
            fig.add_trace(
                go.Scatter(
                    x=sd["Year"],
                    y=sd["Skill Rank"],
                    mode="lines+markers",
                    name=skill,
                    legendgroup=skill,
                    line=dict(color=color_map[skill], width=3, shape="spline"),
                    marker=dict(size=10),
                    visible=visible_flag,
                    hovertemplate=(
                        f"<b>{skill}</b><br>"
                        f"Industry: {ind}<br>"
                        "Year: %{x}<br>"
                        "Rank: %{y}<extra></extra>"
                    )
                )
            )
            
            trace_industry.append(ind)
    
    # Dropdown menu
    buttons = []
    for ind in industries:
        visible = [trace_industry[i] == ind for i in range(len(trace_industry))]
        
        buttons.append(
            dict(
                label=ind,
                method="update",
                args=[
                    {"visible": visible},
                    {"title": f"{country} – Skill rank over time in {ind} (top {top_k} skills)"},
                ],
            )
        )
    
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=0.03,
                xanchor="left",
                y=1.13,
                yanchor="top",
            )
        ]
    )
    
    # Axes & Style
    fig.update_yaxes(autorange="reversed", title="Skill rank (1 = top)")
    fig.update_xaxes(title="Year", tickmode="linear")
    
    fig.update_layout(
        title=f"{country} – Skill rank over time in {initial_industry} (top {top_k} skills)",
        template="simple_white",
        hovermode="closest",
        legend_title_text="Skill",
        margin=dict(l=40, r=200, t=80, b=40),
    )
    
    fig.show()


def plot_skill_rank_flow_all(skill_flow_df, country='Algeria'):
    """
    Plot skill rank evolution over time by industry (all skills).
    
    Parameters:
    -----------
    skill_flow_df : pd.DataFrame
        DataFrame with columns: Country, Industry, Skill, Year, Skill Rank
    country : str
        Country name to filter data for (default: 'Algeria')
    """
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Filter to one country
    df = skill_flow_df[skill_flow_df["Country"] == country].copy()
    
    if df.empty:
        print(f"No data available for {country}")
        return
    
    df = df.sort_values(["Industry", "Year", "Skill Rank"])
    industries = sorted(df["Industry"].unique())
    
    # Global color map: same color for a skill across all industries
    all_skills = sorted(df["Skill"].unique())
    base_colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3
    color_map = {s: base_colors[i % len(base_colors)] for i, s in enumerate(all_skills)}
    
    # Build one trace per (industry, skill)
    fig = go.Figure()
    trace_industries = []  # to know which trace belongs to which industry
    
    initial_industry = industries[0]  # default shown at start
    
    for ind in industries:
        df_ind = df[df["Industry"] == ind]
        
        for skill in sorted(df_ind["Skill"].unique()):
            sd = df_ind[df_ind["Skill"] == skill].sort_values("Year")
            if sd.empty:
                continue
            
            visible_flag = (ind == initial_industry)
            
            fig.add_trace(
                go.Scatter(
                    x=sd["Year"],
                    y=sd["Skill Rank"],
                    mode="lines+markers",
                    name=skill,
                    legendgroup=skill,  # same legend entry per skill across industries
                    line=dict(
                        color=color_map[skill],
                        width=2,
                        shape="spline"   # smoother line
                    ),
                    marker=dict(size=8),
                    hovertemplate=(
                        f"<b>{skill}</b><br>" +
                        f"Industry: {ind}<br>" +
                        "Year: %{x}<br>" +
                        "Rank: %{y}<extra></extra>"
                    ),
                    visible=visible_flag
                )
            )
            trace_industries.append(ind)
    
    # Dropdown to filter by industry
    buttons = []
    for ind in industries:
        visible = [(trace_industries[i] == ind) for i in range(len(trace_industries))]
        
        buttons.append(
            dict(
                label=ind,
                method="update",
                args=[
                    {"visible": visible},
                    {"title": f"{country} – Skill rank evolution in {ind} (all skills)"}
                ]
            )
        )
    
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.02,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ]
    )
    
    # Axes & style
    fig.update_yaxes(
        autorange="reversed",   # 1 at the top
        title="Skill rank (1 = top)"
    )
    fig.update_xaxes(
        title="Year",
        tickmode="linear"
    )
    
    fig.update_layout(
        title=f"{country} – Skill rank evolution in {initial_industry} (all skills)",
        template="simple_white",
        hovermode="closest",
        legend_title_text="Skill",
        margin=dict(l=40, r=260, t=80, b=40)
    )
    
    fig.show()


def plot_women_share_by_peer_groups(gender_country_df):
    """
    Plot women's share of employment over time by peer groups.
    
    Parameters:
    -----------
    gender_country_df : pd.DataFrame
        DataFrame with columns: year_start_employed, country_name, women_perc
        
    Returns:
    --------
    None (shows Bokeh plot)
    """
    from bokeh.io import output_file, show
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, HoverTool, Tabs, TabPanel, Legend, Div, Button, CustomJS
    from bokeh.palettes import Category10
    from bokeh.layouts import column
    import os
    
    try:
        df = gender_country_df.copy()
        df["year_start_employed"] = df["year_start_employed"].astype(int)
        df["women_pct"] = df["women_perc"] * 100
        
        peer_groups = {
            "MENA (regional peers)": [
                "Algeria", "Egypt", "Iraq", "Jordan", "Morocco", "Tunisia"
            ],
            "Non-MENA (structural peers)": [
                "Algeria", "Uzbekistan", "Ecuador", "Peru", "Ghana", "Vietnam", "Colombia"
            ],
            "Non-MENA (aspirational peers)": [
                "Algeria", "Chile", "Kazakhstan", "Malaysia", "Poland", "Romania", "Turkey"
            ]
        }
        
        tabs = []
        
        for group_name, countries in peer_groups.items():
            sub = df[df["country_name"].isin(countries)].copy()
            if sub.empty:
                continue
            
            sub = sub.sort_values(["country_name", "year_start_employed"])
            
            max_pct = sub["women_pct"].max()
            y_max = min(100, max_pct * 1.15)
            
            unique_countries = sub["country_name"].unique().tolist()
            colors = Category10[max(3, min(10, len(unique_countries)))]
            
            p = figure(
                height=430,
                width=780,
                title=f"Share of women in employment over time – {group_name}",
                toolbar_location=None
            )
            
            legend_items = []
            point_renderers = []
            
            # Prepare download data source
            download_data = sub.copy()
            source_download = ColumnDataSource(download_data)
            
            for i, c in enumerate(unique_countries):
                dc = sub[sub["country_name"] == c]
                src = ColumnDataSource(dc)
                color = colors[i % len(colors)]
                
                line_r = p.line(
                    x="year_start_employed",
                    y="women_pct",
                    source=src,
                    line_width=3,
                    color=color,
                    muted_alpha=0.15
                )
                scat_r = p.scatter(
                    x="year_start_employed",
                    y="women_pct",
                    source=src,
                    size=7,
                    color=color
                )
                
                legend_items.append((c, [line_r, scat_r]))
                point_renderers.append(scat_r)
            
            p.add_tools(HoverTool(
                renderers=point_renderers,
                tooltips=[
                    ("Year", "@year_start_employed"),
                    ("% women", "@women_pct{0.0}%")
                ],
                mode="mouse"
            ))
            
            p.xaxis.axis_label = "Year"
            p.yaxis.axis_label = "Women as % of employed"
            p.y_range.start = 0
            p.y_range.end = y_max
            p.xgrid.grid_line_color = None
            p.ygrid.grid_line_alpha = 0.15
            
            legend = Legend(
                items=legend_items,
                location="center",
                orientation="horizontal",
                padding=2,
                spacing=3,
                glyph_width=10,
                glyph_height=10,
                label_text_font_size="9pt",
            )
            legend.ncols = min(5, len(unique_countries))
            legend.click_policy = "mute"
            p.add_layout(legend, "below")
            
            # Add download button
            download_button = Button(label="Download CSV", button_type="primary")
            download_js = CustomJS(args=dict(source=source_download, group=group_name), code="""
                const data = source.data;
                const cols = Object.keys(data);
                const nrows = data[cols[0]].length;
                let csv = cols.join(",") + "\\n";
                for (let i = 0; i < nrows; i++) {
                    let row = [];
                    for (let col of cols) {
                        row.push(data[col][i]);
                    }
                    csv += row.join(",") + "\\n";
                }
                const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
                const link = document.createElement("a");
                link.href = URL.createObjectURL(blob);
                const filename = "women_share_" + group.replace(/ /g, "_") + ".csv";
                link.download = filename;
                link.click();
            """)
            download_button.js_on_click(download_js)
            
            tab_layout = column(download_button, p)
            tabs.append(TabPanel(child=tab_layout, title=group_name))
        
        title_div = Div(
            text="""
            <h2 style='text-align:center; margin:0; font-size:20px;'>
            Women as a Share of Employed LinkedIn Members, by Country and Year (2015–2025)
            </h2>
            """,
            width=800
        )
        
        layout = column(title_div, Tabs(tabs=tabs))
        
        output_file("women_share_by_peer_groups.html")
        show(layout)
        
    except Exception as e:
        print(f"Error creating women share by peer groups plot: {str(e)}")


def plot_women_share_by_industry(gender_flow_df, peer_countries=None):
    """
    Plot women's share of employment over time by industry for specified countries.
    
    Parameters:
    -----------
    gender_flow_df : pd.DataFrame
        DataFrame with columns: country_name, year_start_employed, Industry, women_perc
    peer_countries : list, optional
        List of country names to plot (default: MENA regional peers)
        
    Returns:
    --------
    None (shows Bokeh plot)
    """
    from bokeh.io import output_file, show
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, HoverTool, Tabs, TabPanel, Legend, Button, CustomJS
    from bokeh.palettes import Category10
    from bokeh.layouts import column
    
    try:
        if peer_countries is None:
            peer_countries = ["Algeria", "Egypt", "Iraq", "Jordan", "Morocco", "Tunisia"]
        
        df = gender_flow_df.copy()
        df = df[df["country_name"].isin(peer_countries)].copy()
        df["year_start_employed"] = df["year_start_employed"].astype(int)
        df["women_pct"] = df["women_perc"] * 100
        
        tabs = []
        
        for country in peer_countries:
            sub = df[df["country_name"] == country].copy()
            if sub.empty:
                continue
            
            industries = sub["Industry"].unique().tolist()
            colors = Category10[max(3, min(10, len(industries)))]
            
            p = figure(
                height=430,
                width=780,
                title=f"Share of women in employment over time – {country}",
                toolbar_location=None
            )
            
            scatter_renderers = []
            legend_items = []
            
            # Prepare download data source
            download_data = sub.copy()
            source_download = ColumnDataSource(download_data)
            
            for i, ind in enumerate(industries):
                di = sub[sub["Industry"] == ind].sort_values("year_start_employed")
                src = ColumnDataSource(di)
                color = colors[i % len(colors)]
                
                line_r = p.line(
                    x="year_start_employed",
                    y="women_pct",
                    source=src,
                    line_width=3,
                    color=color,
                    muted_alpha=0.15
                )
                scat_r = p.scatter(
                    x="year_start_employed",
                    y="women_pct",
                    source=src,
                    size=6,
                    color=color
                )
                
                scatter_renderers.append(scat_r)
                legend_items.append((ind, [line_r, scat_r]))
            
            p.add_tools(HoverTool(
                tooltips=[
                    ("Year", "@year_start_employed"),
                    ("% women", "@women_pct{0.0}%")
                ],
                renderers=scatter_renderers,
                mode="mouse"
            ))
            
            p.xaxis.axis_label = "Year"
            p.yaxis.axis_label = "Women as % of employed"
            p.y_range.start = 0
            p.y_range.end = 50
            p.xgrid.grid_line_color = None
            p.ygrid.grid_line_alpha = 0.15
            
            legend = Legend(
                items=legend_items,
                location="center",
                orientation="horizontal",
                padding=2,
                spacing=3,
                glyph_width=10,
                glyph_height=10,
                label_text_font_size="7.5pt",
            )
            p.add_layout(legend, "below")
            legend.ncols = min(4, len(industries))
            legend.click_policy = "mute"
            
            # Add download button
            download_button = Button(label="Download CSV", button_type="primary")
            download_js = CustomJS(args=dict(source=source_download, country=country), code="""
                const data = source.data;
                const cols = Object.keys(data);
                const nrows = data[cols[0]].length;
                let csv = cols.join(",") + "\\n";
                for (let i = 0; i < nrows; i++) {
                    let row = [];
                    for (let col of cols) {
                        row.push(data[col][i]);
                    }
                    csv += row.join(",") + "\\n";
                }
                const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
                const link = document.createElement("a");
                link.href = URL.createObjectURL(blob);
                const filename = "women_share_by_industry_" + country.replace(/ /g, "_") + ".csv";
                link.download = filename;
                link.click();
            """)
            download_button.js_on_click(download_js)
            
            tab_layout = column(download_button, p)
            tabs.append(TabPanel(child=tab_layout, title=country))
        
        output_file("women_share_by_industry.html")
        show(Tabs(tabs=tabs))
        
    except Exception as e:
        print(f"Error creating women share by industry plot: {str(e)}")


def plot_women_stem_by_peer_groups(women_stem_df):
    """
    Plot women's share in STEM occupations over time by peer groups.
    
    Parameters:
    -----------
    women_stem_df : pd.DataFrame
        DataFrame with columns: year_start_employed, country_name, women_perc
        
    Returns:
    --------
    None (shows Bokeh plot)
    """
    from bokeh.io import output_file, show
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, HoverTool, Tabs, TabPanel, Legend, Div, Button, CustomJS
    from bokeh.palettes import Category10
    from bokeh.layouts import column
    
    try:
        df = women_stem_df.copy()
        df["year_start_employed"] = df["year_start_employed"].astype(int)
        df["women_pct"] = df["women_perc"] * 100
        
        peer_groups = {
            "MENA (regional peers)": [
                "Algeria", "Egypt", "Iraq", "Jordan", "Morocco", "Tunisia"
            ],
            "Non-MENA (structural peers)": [
                "Algeria", "Uzbekistan", "Ecuador", "Peru", "Ghana", "Vietnam", "Colombia"
            ],
            "Non-MENA (aspirational peers)": [
                "Algeria", "Chile", "Kazakhstan", "Malaysia", "Poland", "Romania", "Turkey"
            ]
        }
        
        tabs = []
        
        for group_name, countries in peer_groups.items():
            sub = df[df["country_name"].isin(countries)].copy()
            if sub.empty:
                continue
            
            sub = sub.sort_values(["country_name", "year_start_employed"])
            
            max_pct = sub["women_pct"].max()
            y_max = min(100, max_pct * 1.15)
            
            unique_countries = sub["country_name"].unique().tolist()
            colors = Category10[max(3, min(10, len(unique_countries)))]
            
            p = figure(
                height=430,
                width=780,
                title=f"Women in STEM occupations over time – {group_name}",
                toolbar_location=None
            )
            
            legend_items = []
            point_renderers = []
            
            # Prepare download data source
            download_data = sub.copy()
            source_download = ColumnDataSource(download_data)
            
            for i, c in enumerate(unique_countries):
                dc = sub[sub["country_name"] == c]
                src = ColumnDataSource(dc)
                color = colors[i % len(colors)]
                
                line_r = p.line(
                    x="year_start_employed",
                    y="women_pct",
                    source=src,
                    line_width=3,
                    color=color,
                    muted_alpha=0.15
                )
                scat_r = p.scatter(
                    x="year_start_employed",
                    y="women_pct",
                    source=src,
                    size=7,
                    color=color
                )
                
                legend_items.append((c, [line_r, scat_r]))
                point_renderers.append(scat_r)
            
            p.add_tools(HoverTool(
                renderers=point_renderers,
                tooltips=[
                    ("Year", "@year_start_employed"),
                    ("% women", "@women_pct{0.0}%")
                ],
                mode="mouse"
            ))
            
            p.xaxis.axis_label = "Year"
            p.yaxis.axis_label = "Women in STEM as % of employed"
            p.y_range.start = 0
            p.y_range.end = y_max
            p.xgrid.grid_line_color = None
            p.ygrid.grid_line_alpha = 0.15
            
            legend = Legend(
                items=legend_items,
                location="center",
                orientation="horizontal",
                padding=2,
                spacing=3,
                glyph_width=10,
                glyph_height=10,
                label_text_font_size="9pt",
            )
            legend.ncols = min(5, len(unique_countries))
            legend.click_policy = "mute"
            p.add_layout(legend, "below")
            
            # Add download button
            download_button = Button(label="Download CSV", button_type="primary")
            download_js = CustomJS(args=dict(source=source_download, group=group_name), code="""
                const data = source.data;
                const cols = Object.keys(data);
                const nrows = data[cols[0]].length;
                let csv = cols.join(",") + "\\n";
                for (let i = 0; i < nrows; i++) {
                    let row = [];
                    for (let col of cols) {
                        row.push(data[col][i]);
                    }
                    csv += row.join(",") + "\\n";
                }
                const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
                const link = document.createElement("a");
                link.href = URL.createObjectURL(blob);
                const filename = "women_stem_" + group.replace(/ /g, "_") + ".csv";
                link.download = filename;
                link.click();
            """)
            download_button.js_on_click(download_js)
            
            tab_layout = column(download_button, p)
            tabs.append(TabPanel(child=tab_layout, title=group_name))
        
        title_div = Div(
            text="""
            <h2 style='text-align:center; margin:0; font-size:20px;'>
            Women as a Share of Employed LinkedIn Members in STEM Occupations, 
            by Country and Year (2015–2024)
            </h2>
            """,
            width=800
        )
        
        layout = column(title_div, Tabs(tabs=tabs))
        
        output_file("women_stem_by_peer_groups.html")
        show(layout)
        
    except Exception as e:
        print(f"Error creating women STEM by peer groups plot: {str(e)}")


def plot_women_leadership_heatmap(gender_industry_df, country='Algeria'):
    """
    Plot women's representation across leadership levels by industry as a heatmap.
    
    Parameters:
    -----------
    gender_industry_df : pd.DataFrame
        DataFrame with industry and seniority level columns
    country : str
        Country name to filter (default: 'Algeria')
        
    Returns:
    --------
    None (shows Bokeh plot)
    """
    from bokeh.io import output_file, show
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter, LabelSet, Div, Button, CustomJS
    from bokeh.layouts import column
    from bokeh.transform import transform
    from bokeh.palettes import Blues9, Oranges9, Greens9
    import pandas as pd
    
    try:
        entry_cols = ['110.Entry-Level', '120.Experienced', '130.Distinguished']
        mid_cols = ['200.Entry-Level Manager', '210.Experienced Manager', '220.Director-Level']
        lead_cols_all = ['300.Vice President-Level', '310.CXO-Level']
        lead_cols = [c for c in lead_cols_all if c in gender_industry_df.columns]
        
        dz = gender_industry_df[gender_industry_df['Country'] == country].copy()
        dz[entry_cols + mid_cols + lead_cols] = dz[entry_cols + mid_cols + lead_cols].apply(
            pd.to_numeric, errors='coerce'
        )
        
        dz_agg = pd.DataFrame()
        dz_agg['Industry'] = dz['Industry']
        dz_agg['Entry-level'] = dz[entry_cols].mean(axis=1, skipna=True)
        dz_agg['Mid-level management'] = dz[mid_cols].mean(axis=1, skipna=True)
        dz_agg['Leadership'] = dz[lead_cols].mean(axis=1, skipna=True)
        
        df_long = dz_agg.melt(
            id_vars='Industry',
            value_vars=['Entry-level', 'Mid-level management', 'Leadership'],
            var_name='Level',
            value_name='Women_share'
        )
        
        df_long['Women_share'] = pd.to_numeric(df_long['Women_share'], errors='coerce')
        df_long['Women_share_pct'] = df_long['Women_share'] * 100
        df_long['pct_label'] = df_long['Women_share_pct'].round(0).astype('Int64').astype(str) + "%"
        
        order = dz_agg.sort_values('Entry-level', ascending=False)['Industry'].tolist()
        df_long['Industry'] = pd.Categorical(df_long['Industry'], categories=order, ordered=True)
        
        LEVELS = ["Entry-level", "Mid-level management", "Leadership"]
        palette_map = {
            "Entry-level": list(reversed(Blues9)),
            "Mid-level management": list(reversed(Oranges9)),
            "Leadership": list(reversed(Greens9))
        }
        
        def map_to_palette(values, palette):
            vals = values.astype(float)
            vmin, vmax = vals.min(), vals.max()
            if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
                mid_color = palette[len(palette) // 2]
                return [mid_color] * len(vals)
            
            n = len(palette)
            colors = []
            for v in vals:
                if pd.isna(v):
                    colors.append("#ffffff")
                else:
                    t = (v - vmin) / (vmax - vmin)
                    idx = int(round(t * (n - 1)))
                    idx = max(0, min(n - 1, idx))
                    colors.append(palette[idx])
            return colors
        
        df_long["color"] = "#ffffff"
        for lvl in LEVELS:
            mask = df_long["Level"] == lvl
            vals = df_long.loc[mask, "Women_share_pct"]
            pal = palette_map[lvl]
            df_long.loc[mask, "color"] = map_to_palette(vals, pal)
        
        source = ColumnDataSource(df_long)
        
        # Prepare download data
        download_data = dz_agg.copy()
        source_download = ColumnDataSource(download_data)
        
        p = figure(
            x_range=LEVELS,
            y_range=list(reversed(order)),
            height=500,
            width=750,
            toolbar_location=None,
            title=f"Percentage of women by level – {country}"
        )
        
        p.rect(
            x="Level",
            y="Industry",
            width=1,
            height=1,
            source=source,
            line_color=None,
            fill_color="color"
        )
        
        labels = LabelSet(
            x='Level',
            y='Industry',
            text='pct_label',
            source=source,
            text_align='center',
            text_baseline='middle',
            text_color='white'
        )
        p.add_layout(labels)
        
        p.xaxis.axis_label = "Position level"
        p.yaxis.axis_label = ""
        p.axis.major_tick_line_color = None
        p.axis.minor_tick_line_color = None
        p.grid.grid_line_color = None
        
        title_div = Div(
            text=f"<h2 style='text-align:center; margin-bottom:10px;'>Women's Representation Across Leadership Levels by Industry – {country}</h2>"
        )
        
        # Add download button
        download_button = Button(label="Download CSV", button_type="primary")
        download_js = CustomJS(args=dict(source=source_download, country=country), code="""
            const data = source.data;
            const cols = Object.keys(data);
            const nrows = data[cols[0]].length;
            let csv = cols.join(",") + "\\n";
            for (let i = 0; i < nrows; i++) {
                let row = [];
                for (let col of cols) {
                    row.push(data[col][i]);
                }
                csv += row.join(",") + "\\n";
            }
            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            const filename = "women_leadership_" + country.replace(/ /g, "_") + ".csv";
            link.download = filename;
            link.click();
        """)
        download_button.js_on_click(download_js)
        
        layout = column(title_div, download_button, p)
        
        output_file(f"women_leadership_heatmap_{country}.html")
        show(layout)
        
    except Exception as e:
        print(f"Error creating women leadership heatmap: {str(e)}")


def plot_women_seniority_curves(gender_df, countries=None):
    """
    Plot women's representation across seniority levels with slope charts.
    
    Parameters:
    -----------
    gender_df : pd.DataFrame
        DataFrame with seniority level columns by sector
    countries : list, optional
        List of country names to include in tabs (default: ['Algeria', 'United States'])
        
    Returns:
    --------
    None (shows Bokeh plot)
    """
    from bokeh.io import output_file, show
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, HoverTool, Tabs, TabPanel, Div, Button, CustomJS
    from bokeh.layouts import column
    from bokeh.palettes import Category10
    import pandas as pd
    
    try:
        if countries is None:
            countries = ['Algeria', 'United States']
        
        seniority_cols = [
            '110.Entry-Level',
            '120.Experienced',
            '130.Distinguished',
            '200.Entry-Level Manager',
            '210.Experienced Manager',
            '220.Director-Level',
            '300.Vice President-Level',
            '310.CXO-Level'
        ]
        
        entry_col = '110.Entry-Level'
        top_col = '310.CXO-Level'
        
        def make_country_figs(df_country, country_label):
            df_country = df_country.copy()
            
            # Find the sector/industry column - could be 'Sector', 'Industry', or similar
            sector_col = None
            for possible_col in ['Sector', 'Industry', 'sector', 'industry', 'Vertical', 'vertical']:
                if possible_col in df_country.columns:
                    sector_col = possible_col
                    break
            
            if sector_col is None:
                # Find any column that's not a seniority column or seniority_name_display
                non_seniority = [col for col in df_country.columns 
                                if col not in seniority_cols and 'seniority' not in col.lower()]
                if non_seniority:
                    sector_col = non_seniority[0]
                else:
                    raise ValueError(f"Could not identify sector column in DataFrame. Available columns: {df_country.columns.tolist()}")
            
            df_long = df_country.melt(
                id_vars=[sector_col],
                value_vars=seniority_cols,
                var_name='Seniority_Level',
                value_name='Women_Share'
            )
            
            df_long['Seniority_Level'] = pd.Categorical(
                df_long['Seniority_Level'],
                categories=seniority_cols,
                ordered=True
            )
            
            sectors = df_long[sector_col].unique().tolist()
            n_sectors = len(sectors)
            palette = Category10[10] if n_sectors <= 10 else (Category10[10] * ((n_sectors // 10) + 1))
            
            # Prepare download data for full curve
            download_data_full = df_long.copy()
            source_download_full = ColumnDataSource(download_data_full)
            
            p1 = figure(
                x_range=seniority_cols,
                height=400,
                width=800,
                title=f"Women's Share Across Seniority Levels by Sector ({country_label})",
                toolbar_location="above"
            )
            p1.xaxis.major_label_orientation = 0.8
            p1.xaxis.axis_label = "Seniority level"
            p1.yaxis.axis_label = "Women's share"
            
            hover1 = HoverTool(
                tooltips=[("Women's share", "@Women_Share{0.0}")],
                mode="mouse"
            )
            p1.add_tools(hover1)
            
            for i, sector in enumerate(sectors):
                d_sec = df_long[df_long[sector_col] == sector].sort_values('Seniority_Level')
                source = ColumnDataSource(d_sec)
                color = palette[i]
                
                p1.line(
                    x='Seniority_Level',
                    y='Women_Share',
                    source=source,
                    line_width=2,
                    color=color,
                    legend_label=sector
                )
                p1.scatter(
                    x='Seniority_Level',
                    y='Women_Share',
                    source=source,
                    size=6,
                    color=color
                )
            
            p1.legend.location = "top_right"
            p1.legend.click_policy = "hide"
            
            # Download button for full curve
            download_button1 = Button(label="Download CSV (Full Curve)", button_type="primary")
            download_js1 = CustomJS(args=dict(source=source_download_full, country=country_label), code="""
                const data = source.data;
                const cols = Object.keys(data);
                const nrows = data[cols[0]].length;
                let csv = cols.join(",") + "\\n";
                for (let i = 0; i < nrows; i++) {
                    let row = [];
                    for (let col of cols) {
                        row.push(data[col][i]);
                    }
                    csv += row.join(",") + "\\n";
                }
                const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
                const link = document.createElement("a");
                link.href = URL.createObjectURL(blob);
                const filename = "women_seniority_full_" + country.replace(/ /g, "_") + ".csv";
                link.download = filename;
                link.click();
            """)
            download_button1.js_on_click(download_js1)
            
            # Slope chart
            df_slope = (
                df_country
                .loc[:, [sector_col, entry_col, top_col]]
                .melt(
                    id_vars=sector_col,
                    value_vars=[entry_col, top_col],
                    var_name='Seniority_Level',
                    value_name='Women_Share'
                )
            )
            
            level_label_map = {
                entry_col: "Entry level",
                top_col: "CXO-Level"
            }
            df_slope['Level_label'] = df_slope['Seniority_Level'].map(level_label_map)
            
            # Prepare download data for slope
            download_data_slope = df_slope.copy()
            source_download_slope = ColumnDataSource(download_data_slope)
            
            p2 = figure(
                x_range=["Entry level", "CXO-Level"],
                height=350,
                width=800,
                title=f"Drop in Women's Representation: Entry Level → CXO-Level ({country_label})",
                toolbar_location="above"
            )
            p2.xaxis.axis_label = "Seniority level"
            p2.yaxis.axis_label = "Women's share"
            
            hover2 = HoverTool(
                tooltips=[("Women's share", "@Women_Share{0.0}")],
                mode="mouse"
            )
            p2.add_tools(hover2)
            
            for i, sector in enumerate(sectors):
                d_sec = df_slope[df_slope[sector_col] == sector].sort_values('Level_label')
                source = ColumnDataSource(d_sec)
                color = palette[i]
                
                p2.line(
                    x='Level_label',
                    y='Women_Share',
                    source=source,
                    line_width=2,
                    color=color,
                    legend_label=sector
                )
                p2.scatter(
                    x='Level_label',
                    y='Women_Share',
                    source=source,
                    size=6,
                    color=color
                )
            
            p2.legend.location = "top_right"
            p2.legend.click_policy = "hide"
            
            # Download button for slope
            download_button2 = Button(label="Download CSV (Slope Chart)", button_type="primary")
            download_js2 = CustomJS(args=dict(source=source_download_slope, country=country_label), code="""
                const data = source.data;
                const cols = Object.keys(data);
                const nrows = data[cols[0]].length;
                let csv = cols.join(",") + "\\n";
                for (let i = 0; i < nrows; i++) {
                    let row = [];
                    for (let col of cols) {
                        row.push(data[col][i]);
                    }
                    csv += row.join(",") + "\\n";
                }
                const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
                const link = document.createElement("a");
                link.href = URL.createObjectURL(blob);
                const filename = "women_seniority_slope_" + country.replace(/ /g, "_") + ".csv";
                link.download = filename;
                link.click();
            """)
            download_button2.js_on_click(download_js2)
            
            layout_country = column(download_button1, p1, download_button2, p2)
            return layout_country
        
        tabs = []
        for country in countries:
            df_country = gender_df[gender_df['seniority_name_display'].str.contains(country, case=False, na=False)]
            if not df_country.empty:
                layout = make_country_figs(df_country, country)
                tabs.append(TabPanel(child=layout, title=country))
        
        # Add README
        readme_text = """
<h3>Seniority Level Definitions</h3>

<p><strong>Individual Contributor (IC)</strong> roles refer to employees who <em>do not</em> manage others. 
They have no direct reports, although they may lead technical projects or workstreams.</p>

<ul>
<li><strong>Entry-Level (IC – 110):</strong> Early-career employees with limited experience.</li>
<li><strong>Experienced (IC – 120):</strong> Employees with established skills and experience.</li>
<li><strong>Distinguished (IC – 130):</strong> Senior technical experts without managerial duties.</li>
</ul>

<p><strong>Mid-Level Management</strong> includes employees who supervise one or more direct reports.</p>

<ul>
<li><strong>Entry-Level Manager (200):</strong> Leads a small team or supervises limited staff.</li>
<li><strong>Experienced Manager (210):</strong> Manages larger teams or multiple functional units.</li>
<li><strong>Director (220):</strong> Oversees significant departmental planning and execution.</li>
</ul>

<p><strong>Leadership</strong> roles hold strategic responsibility over major business functions 
or the entire organization.</p>

<ul>
<li><strong>Vice President-Level (300):</strong> Senior leaders managing major business units.</li>
<li><strong>C-Suite Executives (310):</strong> Top-level officers such as CEO, COO, CFO, CTO.</li>
</ul>

<hr>
<p><strong>Source:</strong> LinkedIn The State of Women in Leadership.</p>
"""
        readme_div = Div(text=readme_text, width=800)
        tabs.append(TabPanel(child=readme_div, title="README"))
        
        title_div = Div(
            text="<h2 style='text-align:center; margin-bottom:10px;'>Women's Representation Across Leadership Levels</h2>"
        )
        
        final_layout = column(title_div, Tabs(tabs=tabs))
        
        output_file("women_seniority_curves.html")
        show(final_layout)
        
    except Exception as e:
        print(f"Error creating women seniority curves: {str(e)}")
