"""
Visualization functions for air pollution data analysis in Algeria.

This module contains functions for creating visualizations of air pollution data
using matplotlib with optional World Bank styling via wbpyplot package.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from wbpyplot import wb_plot
    WBPYPLOT_AVAILABLE = True
    print("✅ wbpyplot successfully imported")
except ImportError:
    print("⚠️ Warning: wbpyplot not installed. Install with: pip install wbpyplot")
    WBPYPLOT_AVAILABLE = False
    # Create a dummy decorator if wbpyplot is not available
    def wb_plot(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def plot_wb_manual_style(data: pd.DataFrame,
                        x_column: str,
                        y_column: str, 
                        category_column: str,
                        title: str,
                        subtitle: str = "",
                        source_note: str = "Satellite data from Google Earth Engine",
                        figsize: tuple = (12, 8)):
    """
    Create a plot with manual World Bank styling when decorator doesn't work.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set more distinct color palette
    distinct_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'   # Cyan
    ]
    
    # Plot data
    categories = data[category_column].unique()
    
    for i, category in enumerate(categories):
        category_data = data[data[category_column] == category]
        if len(category_data) == 0:
            continue
        
        color = distinct_colors[i % len(distinct_colors)]
        ax.plot(category_data[x_column], 
               category_data[y_column], 
               label=str(category),
               color=color,
               linewidth=2.5)
    
    # Apply World Bank styling manually
    # Left-aligned title
    ax.text(0.02, 0.98, title, transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='top', ha='left')
    
    # Subtitle
    if subtitle:
        ax.text(0.02, 0.92, subtitle, transform=ax.transAxes, 
                fontsize=12, va='top', ha='left', style='italic')
    
    # Source note at bottom
    fig.text(0.02, 0.02, f"Source: {source_note}", 
             fontsize=10, ha='left', va='bottom')
    
    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # Grid
    ax.grid(True, color='#EEEEEE', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Axis labels
    ax.set_xlabel(x_column.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel(y_column.replace('_', ' ').title(), fontsize=11)
    
    # Legend
    if len(categories) > 1:
        ax.legend(loc='upper right', frameon=False)
    
    # Adjust layout to make room for title and source
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    plt.show()
    
    return fig


@wb_plot(
    title="Air Pollution Line Chart",
    subtitle="NO2 Column Number Density over time"
)
def plot_line_chart_by_category_wb(axs, df: pd.DataFrame,
                               x_column: str,
                               y_column: str,
                               category_column: str,
                               **kwargs):
    """
    Create a line chart with World Bank styling using the new wbpyplot decorator.
    This is a simplified version that works with the @wb_plot decorator.
    """
    ax = axs[0]
    
    # Get unique categories
    categories = df[category_column].unique()
    
    # Plot lines for each category
    for i, category in enumerate(categories):
        category_data = df[df[category_column] == category]
        
        if len(category_data) == 0:
            continue
        
        # Plot line
        ax.plot(category_data[x_column], 
               category_data[y_column], 
               label=str(category),
               linewidth=2)
    
    # Add labels
    ax.set_xlabel(x_column.replace('_', ' ').title())
    ax.set_ylabel(y_column.replace('_', ' ').title())
    
    # Add legend if multiple categories
    if len(categories) > 1:
        ax.legend()


# Create a function that returns decorator functions with data embedded
def create_wb_plot_function(data, title, subtitle, note, x_col, y_col, cat_col):
    """Create a wbpyplot decorated function with embedded data"""
    
    @wb_plot(
        title=title,
        subtitle=subtitle,
        note=note
    )
    def plot_function(axs, **kwargs):
        ax = axs[0]
        
        categories = data[cat_col].unique()
        
        for i, category in enumerate(categories):
            category_data = data[data[cat_col] == category]
            if len(category_data) == 0:
                continue
            
            ax.plot(category_data[x_col], 
                   category_data[y_col], 
                   label=str(category),
                   linewidth=2.5)
        
        ax.set_xlabel(x_col.replace('_', ' ').title())
        ax.set_ylabel(y_col.replace('_', ' ').title())
        
        if len(categories) > 1:
            ax.legend()
    
    return plot_function


# World Bank styled plotting function using standard matplotlib subplots
def plot_wb_styled(data: pd.DataFrame,
                   x_column: str,
                   y_column: str, 
                   category_column: str,
                   title: str,
                   subtitle: str = "",
                   source_note: str = "Satellite data from Google Earth Engine",
                   figsize: tuple = (12, 8)):
    """
    Create a plot with World Bank styling using standard matplotlib subplots.
    This maintains accurate axis alignment while applying WB visual style.
    """
    import matplotlib.pyplot as plt
    
    # Create figure and axis using standard matplotlib
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set more distinct color palette
    distinct_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'   # Cyan
    ]
    
    # Plot data
    categories = data[category_column].unique()
    
    for i, category in enumerate(categories):
        category_data = data[data[category_column] == category]
        if len(category_data) == 0:
            continue
        
        color = distinct_colors[i % len(distinct_colors)]
        ax.plot(category_data[x_column], 
               category_data[y_column], 
               label=str(category),
               color=color,
               linewidth=2.5)
    
    # Apply World Bank title styling - LEFT ALIGNED
    # Remove the default title and add custom positioned text
    ax.set_title('')  # Remove default title
    
    # Add left-aligned title at the top
    fig.text(0.125, 0.95, title, 
             fontsize=16, fontweight='bold', 
             ha='left', va='top',
             transform=fig.transFigure)
    
    # Add subtitle if provided
    if subtitle:
        fig.text(0.125, 0.91, subtitle, 
                 fontsize=12, ha='left', va='top',
                 style='italic', color='#666666',
                 transform=fig.transFigure)
    
    # Add source note at bottom left
    fig.text(0.125, 0.02, f"Source: {source_note}", 
             fontsize=10, ha='left', va='bottom',
             color='#666666',
             transform=fig.transFigure)
    
    # Clean up axes - World Bank style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # Grid styling
    ax.grid(True, color='#EEEEEE', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Axis labels
    ax.set_xlabel(x_column.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel(y_column.replace('_', ' ').title(), fontsize=11)
    
    # Legend
    if len(categories) > 1:
        ax.legend(loc='upper right', frameon=False)
    
    # Adjust layout to make room for title and source
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.15)
    
    plt.show()


def plot_wb_styled_dual_axis(data: pd.DataFrame,
                              x_column: str,
                              y1_column: str,
                              y2_column: str,
                              category_column: Optional[str] = None,
                              y1_label: Optional[str] = None,
                              y2_label: Optional[str] = None,
                              title: str = "Dual Axis Chart",
                              subtitle: str = "",
                              source_note: str = "Satellite data from Google Earth Engine",
                              figsize: tuple = (12, 8),
                              y1_color: str = '#1f77b4',
                              y2_color: str = '#ff7f0e'):
    """
    Create a dual-axis plot with World Bank styling using standard matplotlib.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe with data to plot
    x_column : str
        Column name for x-axis (shared by both series)
    y1_column : str
        Column name for left y-axis (primary axis)
    y2_column : str
        Column name for right y-axis (secondary axis)
    category_column : str, optional
        Column name for grouping data (e.g., for multiple lines per axis)
    y1_label : str, optional
        Custom label for left y-axis (defaults to column name)
    y2_label : str, optional
        Custom label for right y-axis (defaults to column name)
    title : str
        Main title for the chart
    subtitle : str
        Subtitle text
    source_note : str
        Source attribution text
    figsize : tuple
        Figure size (width, height)
    y1_color : str
        Color for left axis and its data
    y2_color : str
        Color for right axis and its data
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object
    """
    import matplotlib.pyplot as plt
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Create secondary axis
    ax2 = ax1.twinx()
    
    # Default labels if not provided
    if y1_label is None:
        y1_label = y1_column.replace('_', ' ').title()
    if y2_label is None:
        y2_label = y2_column.replace('_', ' ').title()
    
    # Plot data on first axis
    if category_column and category_column in data.columns:
        categories = data[category_column].unique()
        for i, category in enumerate(categories):
            category_data = data[data[category_column] == category]
            if len(category_data) == 0:
                continue
            ax1.plot(category_data[x_column], 
                    category_data[y1_column], 
                    label=f"{y1_label} - {category}",
                    color=y1_color,
                    linewidth=2.5,
                    alpha=0.7 if len(categories) > 1 else 1.0)
    else:
        ax1.plot(data[x_column], 
                data[y1_column], 
                label=y1_label,
                color=y1_color,
                linewidth=2.5)
    
    # Plot data on second axis
    if category_column and category_column in data.columns:
        categories = data[category_column].unique()
        for i, category in enumerate(categories):
            category_data = data[data[category_column] == category]
            if len(category_data) == 0:
                continue
            ax2.plot(category_data[x_column], 
                    category_data[y2_column], 
                    label=f"{y2_label} - {category}",
                    color=y2_color,
                    linewidth=2.5,
                    alpha=0.7 if len(categories) > 1 else 1.0)
    else:
        ax2.plot(data[x_column], 
                data[y2_column], 
                label=y2_label,
                color=y2_color,
                linewidth=2.5)
    
    # Apply World Bank title styling - LEFT ALIGNED
    fig.text(0.125, 0.95, title, 
             fontsize=16, fontweight='bold', 
             ha='left', va='top',
             transform=fig.transFigure)
    
    # Add subtitle if provided
    if subtitle:
        fig.text(0.125, 0.91, subtitle, 
                 fontsize=12, ha='left', va='top',
                 style='italic', color='#666666',
                 transform=fig.transFigure)
    
    # Add source note at bottom left
    fig.text(0.125, 0.02, f"Source: {source_note}", 
             fontsize=10, ha='left', va='bottom',
             color='#666666',
             transform=fig.transFigure)
    
    # Style the axes
    ax1.set_xlabel(x_column.replace('_', ' ').title(), fontsize=11)
    ax1.set_ylabel(y1_label, fontsize=11, color=y1_color)
    ax2.set_ylabel(y2_label, fontsize=11, color=y2_color)
    
    # Color the axis labels and ticks
    ax1.tick_params(axis='y', labelcolor=y1_color)
    ax2.tick_params(axis='y', labelcolor=y2_color)
    
    # Clean up axes - World Bank style
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_color(y1_color)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_color('#CCCCCC')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color(y2_color)
    ax2.spines['right'].set_linewidth(1.5)
    
    # Grid styling (only on primary axis)
    ax1.grid(True, color='#EEEEEE', linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
               loc='upper left', frameon=False, fontsize=10)
    
    # Adjust layout to make room for title and source
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.15)
    
    plt.show()
    
    return fig
    
    return fig


@wb_plot(
    title="Air Pollution Line Chart",
    subtitle="NO2 Column Number Density over time"
)
def plot_line_chart_by_category_wb(axs, df: pd.DataFrame,
                               x_column: str,
                               y_column: str,
                               category_column: str,
                               **kwargs):
    """
    Create a line chart with World Bank styling using the new wbpyplot decorator.
    This is a simplified version that works with the @wb_plot decorator.
    """
    ax = axs[0]
    
    # Get unique categories
    categories = df[category_column].unique()
    
    # Plot lines for each category
    for i, category in enumerate(categories):
        category_data = df[df[category_column] == category]
        
        if len(category_data) == 0:
            continue
        
        # Plot line
        ax.plot(category_data[x_column], 
               category_data[y_column], 
               label=str(category),
               linewidth=2)
    
    # Add labels
    ax.set_xlabel(x_column.replace('_', ' ').title())
    ax.set_ylabel(y_column.replace('_', ' ').title())
    
    # Add legend if multiple categories
    if len(categories) > 1:
        ax.legend()


# Create specific functions for different chart types with appropriate titles
@wb_plot(
    title="Average NO2 Column Number Density in Algeria Over Time",
    subtitle="Annual trends by administrative region",
    note=[("Source:", "Satellite data from Google Earth Engine")]
)
def plot_no2_annual_trends_wb(axs, df: pd.DataFrame,
                             x_column: str = 'year',
                             y_column: str = 'NO2_mean',
                             category_column: str = 'NAME_0',
                             **kwargs):
    """World Bank styled plot for annual NO2 trends."""
    ax = axs[0]
    
    categories = df[category_column].unique()
    
    for i, category in enumerate(categories):
        category_data = df[df[category_column] == category]
        if len(category_data) == 0:
            continue
        
        ax.plot(category_data[x_column], 
               category_data[y_column], 
               label=str(category),
               linewidth=2.5)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('NO2 Column Number Density (mol/m²)')
    
    if len(categories) > 1:
        ax.legend()


@wb_plot(
    title="Monthly NO2 Values in Algeria",
    subtitle="Time series showing seasonal patterns",
    note=[("Source:", "Satellite data from Google Earth Engine")]
)
def plot_no2_monthly_trends_wb(axs, df: pd.DataFrame,
                              x_column: str = 'start_date',
                              y_column: str = 'NO2_mean',
                              category_column: str = 'NAME_0',
                              **kwargs):
    """World Bank styled plot for monthly NO2 trends."""
    ax = axs[0]
    
    categories = df[category_column].unique()
    
    for i, category in enumerate(categories):
        category_data = df[df[category_column] == category]
        if len(category_data) == 0:
            continue
        
        ax.plot(category_data[x_column], 
               category_data[y_column], 
               label=str(category),
               linewidth=2.5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('NO2 Column Number Density (mol/m²)')
    
    if len(categories) > 1:
        ax.legend()


def plot_line_chart_by_category(df: pd.DataFrame,
                               x_column: str,
                               y_column: str,
                               category_column: str,
                               title: Optional[str] = None,
                               x_label: Optional[str] = None,
                               y_label: Optional[str] = None,
                               date_format: Optional[str] = None,
                               figsize: tuple = (12, 8),
                               color_palette: Optional[str] = None,
                               show_legend: bool = True,
                               save_path: Optional[str] = None,
                               **kwargs) -> plt.Figure:
    """
    Create a line chart with separate lines for each category using wbpyplot styling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the data to plot
    x_column : str
        Column name for x-axis (typically date/time)
    y_column : str
        Column name for y-axis (values to plot)
    category_column : str
        Column name for category to split lines by
    title : str, optional
        Chart title
    x_label : str, optional
        X-axis label
    y_label : str, optional
        Y-axis label
    date_format : str, optional
        Date format for x-axis labels (e.g., '%Y-%m', '%b %Y')
    figsize : tuple
        Figure size (width, height)
    color_palette : str, optional
        Color palette to use ('wb', 'viridis', 'Set1', etc.)
    show_legend : bool
        Whether to show legend
    save_path : str, optional
        Path to save the figure
    **kwargs
        Additional arguments passed to the plotting function
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    
    # Validate inputs
    required_columns = [x_column, y_column, category_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in dataframe: {missing_columns}")
    
    # Create a copy to avoid modifying original data
    plot_df = df.copy()
    
    # Convert date column if it's not already datetime
    if pd.api.types.is_object_dtype(plot_df[x_column]):
        try:
            plot_df[x_column] = pd.to_datetime(plot_df[x_column])
        except:
            pass  # Keep as is if conversion fails
    
    # Sort by x_column for proper line plotting
    plot_df = plot_df.sort_values(x_column)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use seaborn styling as fallback
    sns.set_style("whitegrid")
    if color_palette:
        colors = plt.cm.get_cmap(color_palette)
    else:
        colors = plt.cm.Set1
    
    # Get unique categories
    categories = plot_df[category_column].unique()
    
    # Plot lines for each category
    for i, category in enumerate(categories):
        category_data = plot_df[plot_df[category_column] == category]
        
        if len(category_data) == 0:
            continue
            
        # Get color
        if hasattr(colors, '__call__'):
            color = colors(i / len(categories))
        else:
            color = colors[i % len(colors)]
        
        # Plot line
        ax.plot(category_data[x_column], 
               category_data[y_column], 
               label=str(category),
               color=color,
               linewidth=2,
               **kwargs)
    
    # Customize the plot
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    if x_label:
        ax.set_xlabel(x_label, fontsize=12)
    else:
        ax.set_xlabel(x_column.replace('_', ' ').title(), fontsize=12)
    
    if y_label:
        ax.set_ylabel(y_label, fontsize=12)
    else:
        ax.set_ylabel(y_column.replace('_', ' ').title(), fontsize=12)
    
    # Format date axis if applicable
    if pd.api.types.is_datetime64_any_dtype(plot_df[x_column]) and date_format:
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        plt.xticks(rotation=45)
    
    # Add legend
    if show_legend and len(categories) > 1:
        ax.legend(title=category_column.replace('_', ' ').title(),
                 bbox_to_anchor=(1.05, 1), 
                 loc='upper left',
                 frameon=True)
    
    # Apply basic styling
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_multiple_pollutants_by_region(df: pd.DataFrame,
                                     date_column: str,
                                     pollutant_columns: List[str],
                                     region_column: str,
                                     regions_to_plot: Optional[List[str]] = None,
                                     title_prefix: str = "Air Pollution Trends",
                                     figsize: tuple = (15, 10),
                                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Create subplots showing multiple pollutants across different regions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    date_column : str
        Column name for date/time
    pollutant_columns : List[str]
        List of pollutant columns to plot
    region_column : str
        Column name for regions
    regions_to_plot : List[str], optional
        Specific regions to include (if None, uses all)
    title_prefix : str
        Prefix for subplot titles
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    
    # Filter regions if specified
    if regions_to_plot:
        df = df[df[region_column].isin(regions_to_plot)]
    
    # Calculate subplot dimensions
    n_pollutants = len(pollutant_columns)
    n_cols = min(2, n_pollutants)
    n_rows = (n_pollutants + n_cols - 1) // n_cols
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    sns.set_style("whitegrid")
    
    # Ensure axes is always a list
    if n_pollutants == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Plot each pollutant
    for i, pollutant in enumerate(pollutant_columns):
        ax = axes[i]
        
        # Create line chart for this pollutant
        plot_data = df[[date_column, pollutant, region_column]].dropna()
        
        if len(plot_data) == 0:
            ax.text(0.5, 0.5, f'No data available for {pollutant}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f"{title_prefix}: {pollutant}")
            continue
        
        # Plot lines for each region
        regions = plot_data[region_column].unique()
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(regions)))
        
        for j, region in enumerate(regions):
            region_data = plot_data[plot_data[region_column] == region]
            color = colors[j]
            
            ax.plot(region_data[date_column], 
                   region_data[pollutant],
                   label=region,
                   color=color,
                   linewidth=2)
        
        # Customize subplot
        ax.set_title(f"{title_prefix}: {pollutant}", fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel(pollutant.replace('_', ' ').title())
        
        if len(regions) > 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Apply styling
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide empty subplots
    for i in range(n_pollutants, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    return fig


def create_air_quality_dashboard(df: pd.DataFrame,
                               date_column: str,
                               pollutant_columns: List[str],
                               region_column: str,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comprehensive air quality dashboard.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    date_column : str
        Column name for date/time
    pollutant_columns : List[str]
        List of pollutant columns
    region_column : str
        Column name for regions
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    
    fig = plt.figure(figsize=(20, 12))
    sns.set_style("whitegrid")
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main time series plot (top row, spans all columns)
    ax_main = fig.add_subplot(gs[0, :])
    
    # Individual pollutant plots (bottom two rows)
    axes_individual = []
    for i in range(min(6, len(pollutant_columns))):
        row = 1 + i // 3
        col = i % 3
        axes_individual.append(fig.add_subplot(gs[row, col]))
    
    # Main plot: Average of all pollutants by region
    main_data = df.groupby([date_column, region_column])[pollutant_columns].mean().reset_index()
    main_data['avg_pollution'] = main_data[pollutant_columns].mean(axis=1)
    
    regions = main_data[region_column].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(regions)))
    
    for i, region in enumerate(regions):
        region_data = main_data[main_data[region_column] == region]
        color = colors[i]
        
        ax_main.plot(region_data[date_column], 
                    region_data['avg_pollution'],
                    label=region,
                    color=color,
                    linewidth=3)
    
    ax_main.set_title('Average Air Pollution Levels by Region', fontsize=16, fontweight='bold', pad=20)
    ax_main.set_xlabel('Date')
    ax_main.set_ylabel('Average Pollution Level')
    ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Individual pollutant plots
    for i, (pollutant, ax) in enumerate(zip(pollutant_columns[:6], axes_individual)):
        pollutant_data = df[[date_column, pollutant, region_column]].dropna()
        
        if len(pollutant_data) == 0:
            ax.text(0.5, 0.5, f'No data for {pollutant}', 
                   transform=ax.transAxes, ha='center', va='center')
            continue
        
        # Plot average across all regions
        avg_data = pollutant_data.groupby(date_column)[pollutant].mean().reset_index()
        
        color = colors[i]
        
        ax.plot(avg_data[date_column], avg_data[pollutant], 
               color=color, linewidth=2)
        ax.set_title(pollutant.replace('_', ' ').title(), fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Level')
    
    # Apply styling to all subplots
    for ax in [ax_main] + axes_individual:
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Dashboard saved to: {save_path}")
    
    return fig


def plot_comparative_maps_over_time(gdf,
                                  value_column: str,
                                  category_column: str,
                                  title: str = "Comparative Maps",
                                  subtitle: str = "",
                                  source_note: str = "Source: Satellite data from Google Earth Engine",
                                  cmap: str = 'YlOrRd',
                                  figsize_per_map: tuple = (6, 6),
                                  max_cols: int = 3,
                                  show_colorbar: bool = True,
                                  show_legend: bool = False,
                                  boundary_color: str = 'white',
                                  boundary_linewidth: float = 0.5,
                                  save_path: Optional[str] = None,
                                  **kwargs) -> plt.Figure:
    """
    Create comparative maps showing spatial data across different categories with World Bank styling.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input geodataframe with geometry and data columns
    value_column : str
        Column name containing the values to map (e.g., pollution levels)
    category_column : str
        Column name containing categories to create separate maps for (e.g., year, month, region)
    title : str
        Main title for the figure
    subtitle : str
        Subtitle text
    source_note : str
        Source attribution text
    cmap : str
        Colormap to use for the maps
    figsize_per_map : tuple
        Size of each individual map (width, height)
    max_cols : int
        Maximum number of columns in the subplot grid
    show_colorbar : bool
        Whether to show colorbar for each map
    show_legend : bool
        Whether to show legend (ignored - only colorbar shown)
    boundary_color : str
        Color for polygon boundaries
    boundary_linewidth : float
        Width of polygon boundaries
    save_path : str, optional
        Path to save the figure
    **kwargs
        Additional arguments passed to the GeoDataFrame plot function
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    
    # Import geopandas here to avoid dependency issues
    try:
        import geopandas as gpd
        import numpy as np
    except ImportError:
        raise ImportError("geopandas and numpy are required for mapping functions")
    
    # Validate inputs
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise ValueError("Input must be a GeoDataFrame")
    
    required_columns = [value_column, category_column]
    missing_columns = [col for col in required_columns if col not in gdf.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in GeoDataFrame: {missing_columns}")
    
    # Get unique categories and sort them
    categories = sorted(gdf[category_column].unique())
    n_categories = len(categories)
    
    if n_categories == 0:
        raise ValueError(f"No categories found in column '{category_column}'")
    
    # Calculate subplot layout
    n_cols = min(max_cols, n_categories)
    n_rows = (n_categories + n_cols - 1) // n_cols
    
    # Calculate figure size with extra space for title and source
    map_width, map_height = figsize_per_map
    total_width = n_cols * map_width
    total_height = n_rows * map_height + 2.0  # Extra space for title/subtitle/source
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(total_width, total_height))
    
    # Ensure axes is always a 2D array for consistent indexing
    if n_categories == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Calculate global min/max for consistent color scaling
    global_min = gdf[value_column].min()
    global_max = gdf[value_column].max()
    
    # Create maps for each category
    for i, category in enumerate(categories):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Filter data for this category
        category_data = gdf[gdf[category_column] == category].copy()
        
        if len(category_data) == 0:
            ax.text(0.5, 0.5, f'No data for {category}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f"{category}", fontsize=12, fontweight='bold', pad=10)
            ax.axis('off')
            continue
        
        # Plot the map
        category_data.plot(
            column=value_column,
            cmap=cmap,
            ax=ax,
            edgecolor=boundary_color,
            linewidth=boundary_linewidth,
            vmin=global_min,
            vmax=global_max,
            legend=False,  # Don't show legend, we'll use colorbar instead
            **kwargs
        )
        
        # Customize the subplot with clean styling
        ax.set_title(f"{category}", fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')
        
        # Remove any axis spines
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Hide empty subplots
    for i in range(n_categories, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    # Add a single colorbar for all maps
    if show_colorbar and n_categories > 0:
        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=global_min, vmax=global_max))
        sm.set_array([])
        
        # Position colorbar on the right side
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.6])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(value_column.replace('_', ' ').title(), rotation=270, labelpad=20, fontsize=11)
        
        # Clean up colorbar styling
        cbar.ax.tick_params(labelsize=10)
    
    # Adjust layout to make room for title and source
    plt.tight_layout()
    if show_colorbar:
        plt.subplots_adjust(top=0.85, bottom=0.1, right=0.9)
    else:
        plt.subplots_adjust(top=0.85, bottom=0.1)
    
    # Add World Bank style title and subtitle (left-aligned, large)
    if title:
        fig.text(0.02, 0.95, title, fontsize=18, fontweight='bold', 
                ha='left', va='top', transform=fig.transFigure)
    
    if subtitle:
        fig.text(0.02, 0.91, subtitle, fontsize=14, 
                ha='left', va='top', transform=fig.transFigure, 
                style='italic', color='#666666')
    
    # Add source note at the bottom left
    if source_note:
        fig.text(0.02, 0.02, source_note, fontsize=10, 
                ha='left', va='bottom', transform=fig.transFigure,
                color='#666666')
    
    # Set figure background to white
    fig.patch.set_facecolor('white')
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Comparative maps saved to: {save_path}")
    
    return fig


def plot_dual_axis_subplots_by_category(
    data: pd.DataFrame,
    category_column: str,
    x_column: str,
    y1_column: str,
    y2_column: str,
    y1_label: str = 'Y1 Value',
    y2_label: str = 'Y2 Value',
    title: str = 'Dual Axis Subplots',
    subtitle: str = '',
    source_note: str = 'Source: Data Analysis',
    y1_color: str = '#1f77b4',
    y2_color: str = '#ff7f0e',
    n_cols: int = 3,
    figsize_per_plot: tuple = (6, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create subplots with dual y-axes for each category in the data.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data to plot
    category_column : str
        Column name to create subplots for (e.g., 'NAME_1' for provinces)
    x_column : str
        Column name for x-axis (typically 'year' or 'date')
    y1_column : str
        Column name for primary y-axis (left side)
    y2_column : str
        Column name for secondary y-axis (right side)
    y1_label : str, optional
        Label for primary y-axis
    y2_label : str, optional
        Label for secondary y-axis
    title : str, optional
        Main title for the entire figure
    subtitle : str, optional
        Subtitle for the figure
    source_note : str, optional
        Source note to display at bottom
    y1_color : str, optional
        Color for primary y-axis line and labels (default: World Bank Blue)
    y2_color : str, optional
        Color for secondary y-axis line and labels (default: World Bank Orange)
    n_cols : int, optional
        Number of columns in subplot grid (default: 3)
    figsize_per_plot : tuple, optional
        Size of each individual subplot (width, height)
    save_path : Optional[str], optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The created matplotlib figure
        
    Examples
    --------
    >>> plot_dual_axis_subplots_by_category(
    ...     data=merged_data,
    ...     category_column='NAME_1',
    ...     x_column='year',
    ...     y1_column='NO2_sum',
    ...     y2_column='ntl_gf_10km_sum',
    ...     y1_label='NO2 Concentration (mol/m²)',
    ...     y2_label='Nighttime Lights (Sum)',
    ...     title='NO2 vs Nighttime Lights by Province'
    ... )
    """
    import math
    
    # Get unique categories and sort them
    categories = sorted(data[category_column].unique())
    n_categories = len(categories)
    
    if n_categories == 0:
        print("Warning: No categories found in data")
        return None
    
    print(f"Creating subplots for {n_categories} categories")
    
    # Calculate subplot grid
    n_cols = min(n_cols, n_categories)
    n_rows = math.ceil(n_categories / n_cols)
    
    # Calculate total figure size
    fig_width = figsize_per_plot[0] * n_cols
    fig_height = figsize_per_plot[1] * n_rows
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # Flatten axes array for easier iteration
    if n_categories == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes])
    
    # Store legend handles and labels from first subplot
    legend_lines = None
    legend_labels = None
    
    # Create subplot for each category
    for idx, category in enumerate(categories):
        ax1 = axes[idx]
        
        # Filter data for this category
        category_data = data[data[category_column] == category].sort_values(x_column)
        
        if len(category_data) == 0:
            ax1.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title(str(category), fontsize=11, fontweight='bold', pad=10)
            continue
        
        # Plot primary y-axis data (left)
        line1 = ax1.plot(category_data[x_column], category_data[y1_column], 
                        color=y1_color, marker='o', linewidth=2, markersize=6, 
                        label=y1_label)
        ax1.set_xlabel('Year', fontsize=10, fontweight='bold')
        ax1.set_ylabel(y1_label, fontsize=9, color=y1_color, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=y1_color, labelsize=9)
        ax1.tick_params(axis='x', labelsize=9)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.set_axisbelow(True)
        
        # Clean up spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Create secondary y-axis (right)
        ax2 = ax1.twinx()
        line2 = ax2.plot(category_data[x_column], category_data[y2_column], 
                        color=y2_color, marker='s', linewidth=2, markersize=6, 
                        label=y2_label)
        ax2.set_ylabel(y2_label, fontsize=9, color=y2_color, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=y2_color, labelsize=9)
        
        # Clean up right spine
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        
        # Set title with category name
        ax1.set_title(str(category), fontsize=11, fontweight='bold', pad=10)
        
        # Store legend info from first subplot only
        if idx == 0:
            lines = line1 + line2
            legend_lines = lines
            legend_labels = [l.get_label() for l in lines]
        
        # Format x-axis to show integer years if applicable
        if category_data[x_column].dtype in ['int64', 'int32']:
            ax1.set_xticks(category_data[x_column].unique())
    
    # Hide unused subplots and add legend to the last one
    for idx in range(n_categories, len(axes)):
        if idx == len(axes) - 1 and legend_lines and legend_labels:
            # Use the last subplot for the legend
            axes[idx].axis('off')
            axes[idx].legend(legend_lines, legend_labels, 
                           loc='center', 
                           fontsize=11, 
                           frameon=True,
                           framealpha=0.95,
                           edgecolor='#CCCCCC',
                           title='Indicators',
                           title_fontsize=12)
        else:
            axes[idx].axis('off')
    
    # Add overall title and subtitle (World Bank style - left aligned)
    if title:
        fig.text(0.02, 0.98, title, 
                fontsize=16, fontweight='bold', 
                ha='left', va='top', transform=fig.transFigure)
    
    if subtitle:
        fig.text(0.02, 0.94, subtitle, 
                fontsize=12, ha='left', va='top',
                style='italic', color='#666666',
                transform=fig.transFigure)
    
    # Add source note at bottom
    if source_note:
        fig.text(0.02, 0.01, source_note, 
                ha='left', va='bottom', fontsize=9, 
                style='italic', color='#666666',
                transform=fig.transFigure)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Set white background
    fig.patch.set_facecolor('white')
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Subplots saved to: {save_path}")
    
    plt.show()
    return fig



