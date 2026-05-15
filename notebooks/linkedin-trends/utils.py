"""
Utility functions for LinkedIn data processing and analysis.
"""

import pandas as pd


def extract_lhr_data_for_countries(sa_file, nonsa_file, countries_list, output_path):
    """
    Extract LHR data (SA and Non-SA) for specified countries and save to processed folder.
    
    Parameters:
    -----------
    sa_file : str
        Path to the Excel file containing seasonally adjusted LHR data
    nonsa_file : str
        Path to the Excel file containing non-seasonally adjusted LHR data
    countries_list : list
        List of country names to extract
    output_path : str
        Path where the combined CSV file should be saved
    
    Returns:
    --------
    pd.DataFrame
        Combined dataset with LHR and LHR_SA for all countries
    """
    
    # Load SA data
    print("Loading Seasonally Adjusted data...")
    sa_data = pd.read_excel(sa_file, sheet_name="2A - LHR SA by Ctry", header=3)
    sa_cleaned = sa_data.rename(columns={
        'Unnamed: 1': 'Month',
        'Unnamed: 2': 'Country',
        'Unnamed: 3': 'LHR_SA'
    })
    sa_cleaned = sa_cleaned.dropna(subset=['Month', 'Country', 'LHR_SA'])
    sa_cleaned = sa_cleaned.iloc[1:]
    sa_cleaned = sa_cleaned.drop(columns=['Unnamed: 0'])
    sa_cleaned["Country"] = sa_cleaned["Country"].str.strip()
    sa_cleaned["Country"] = sa_cleaned["Country"].replace({
        "Turkey": "Turkiye",
        "Türkiye": "Turkiye"
    })
    
    # Load Non-SA data
    print("Loading Non-Seasonally Adjusted data...")
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
    nonsa_cleaned["Country"] = nonsa_cleaned["Country"].replace({
        "Turkey": "Turkiye",
        "Türkiye": "Turkiye"
    })
    
    # Filter for countries of interest
    sa_filtered = sa_cleaned[sa_cleaned['Country'].isin(countries_list)].copy()
    nonsa_filtered = nonsa_cleaned[nonsa_cleaned['Country'].isin(countries_list)].copy()
    
    # Merge SA and Non-SA data
    combined = pd.merge(
        nonsa_filtered[['Month', 'Country', 'LHR']],
        sa_filtered[['Month', 'Country', 'LHR_SA']],
        on=['Month', 'Country'],
        how='outer'
    )
    
    # Sort by Country and Month
    combined['Month'] = pd.to_datetime(combined['Month'])
    combined = combined.sort_values(['Country', 'Month'])
    
    # Save to CSV
    combined.to_csv(output_path, index=False)
    print(f"\nData saved to: {output_path}")
    print(f"Shape: {combined.shape}")
    print(f"Countries included: {sorted(combined['Country'].unique())}")
    print(f"Date range: {combined['Month'].min()} to {combined['Month'].max()}")
    
    # Show summary statistics
    print("\nSummary statistics:")
    print(combined.groupby('Country').agg({
        'LHR': ['count', 'mean', 'std', 'min', 'max'],
        'LHR_SA': ['count', 'mean', 'std', 'min', 'max']
    }).round(2))
    
    return combined


def export_industry_metrics(excel_file, country='Algeria', output_path='industry_performance_metrics.csv'):
    """
    Export industry performance metrics to CSV file.
    
    Parameters:
    -----------
    excel_file : str
        Path to the Excel file containing industry data
    country : str
        Country name to filter data for (default: 'Algeria')
    output_path : str
        Path where the CSV file should be saved
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with industry metrics including averages, volatility, and trends
    """
    from visuals import clean_linkedin_data_industry
    
    try:
        # Clean the data
        df = clean_linkedin_data_industry(excel_file)
        
        if df is None:
            return None
            
        # Filter data for specified country
        bd_data = df[df['Country'] == country].copy()
        bd_data['Month'] = pd.to_datetime(bd_data['Month'])
        
        # Calculate metrics for each industry
        metrics_list = []
        for industry in bd_data['Industry'].unique():
            industry_data = bd_data[bd_data['Industry'] == industry]
            
            # Calculate various metrics
            metrics = {
                'Industry': industry,
                'Average_LHR': industry_data['LHR (YOY)'].mean(),
                'Max_LHR': industry_data['LHR (YOY)'].max(),
                'Min_LHR': industry_data['LHR (YOY)'].min(),
                'Volatility': industry_data['LHR (YOY)'].std(),
                'Latest_LHR': industry_data.loc[industry_data['Month'].idxmax(), 'LHR (YOY)'],
                'Growth_Trend': industry_data['LHR (YOY)'].iloc[-1] - industry_data['LHR (YOY)'].iloc[0],
                'Start_Date': industry_data['Month'].min().strftime('%Y-%m'),
                'End_Date': industry_data['Month'].max().strftime('%Y-%m')
            }
            metrics_list.append(metrics)
        
        # Create DataFrame and sort by Average LHR
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df = metrics_df.sort_values('Average_LHR', ascending=False)
        
        # Add category labels
        n_industries = len(metrics_df)
        categories = ['Leading Industries'] * 5 + \
                    ['Growing Industries'] * 5 + \
                    ['Transitioning Industries'] * 5 + \
                    ['Emerging Industries'] * 5
        metrics_df['Category'] = categories[:n_industries]
        
        # Reorder columns to put Category first
        cols = ['Category'] + [col for col in metrics_df.columns if col != 'Category']
        metrics_df = metrics_df[cols]
        
        # Export to CSV
        metrics_df.to_csv(output_path, index=False)
        print(f"Metrics exported to {output_path}")
        print(f"Countries analyzed: {country}")
        print(f"Number of industries: {len(metrics_df)}")
        
        return metrics_df
        
    except Exception as e:
        print(f"Error exporting metrics: {str(e)}")
        return None


def create_industry_performance_dict(excel_file):
    """
    Create a dictionary with industry performance metrics split into 4 quartiles.
    
    Parameters:
    -----------
    excel_file : str
        Path to the Excel file containing industry data
        
    Returns:
    --------
    dict
        Dictionary with 4 keys (leading_industries, growing_industries, 
        transitioning_industries, emerging_industries), each containing
        industry metrics
    """
    from visuals import clean_linkedin_data_industry
    import os
    
    try:
        # Clean data
        df = clean_linkedin_data_industry(excel_file)
        
        if df is None:
            return None
        
        # Get country from environment or default to Algeria
        country = os.environ.get('COUNTRY', 'Algeria')
            
        # Filter data for country and convert Month to datetime
        bd_data = df[df['Country'] == country].copy()
        bd_data['Month'] = pd.to_datetime(bd_data['Month'])
        
        # Initialize performance dictionary with 4 quartiles
        performance_dict = {
            'leading_industries': {},      # Top 5 (1st quartile)
            'growing_industries': {},      # Next 5 (2nd quartile)
            'transitioning_industries': {}, # Next 5 (3rd quartile)
            'emerging_industries': {}      # Last 5 (4th quartile)
        }
        
        # Calculate metrics for each industry
        industry_metrics = {}
        for industry in bd_data['Industry'].unique():
            industry_data = bd_data[bd_data['Industry'] == industry]
            
            # Calculate industry metrics
            metrics = {
                'avg_lhr': industry_data['LHR (YOY)'].mean(),
                'max_lhr': industry_data['LHR (YOY)'].max(),
                'min_lhr': industry_data['LHR (YOY)'].min(),
                'latest_lhr': industry_data.loc[industry_data['Month'].idxmax(), 'LHR (YOY)'],
                'start_date': industry_data['Month'].min().strftime('%Y-%m'),
                'end_date': industry_data['Month'].max().strftime('%Y-%m'),
                'trend': industry_data['LHR (YOY)'].iloc[-1] - industry_data['LHR (YOY)'].iloc[0],
                'volatility': industry_data['LHR (YOY)'].std()
            }
            industry_metrics[industry] = metrics
        
        # Sort industries by average LHR
        sorted_industries = sorted(industry_metrics.items(), 
                                 key=lambda x: x[1]['avg_lhr'], 
                                 reverse=True)
        
        # Split into 4 groups of 5 industries each
        for i, (industry, metrics) in enumerate(sorted_industries):
            if i < 5:  # First 5 industries
                performance_dict['leading_industries'][industry] = metrics
            elif i < 10:  # Next 5 industries
                performance_dict['growing_industries'][industry] = metrics
            elif i < 15:  # Next 5 industries
                performance_dict['transitioning_industries'][industry] = metrics
            else:  # Last 5 industries
                performance_dict['emerging_industries'][industry] = metrics
        
        return performance_dict
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def clean_skills_data(skills_file):
    """
    Clean and prepare skills genome data.
    
    Parameters:
    -----------
    skills_file : str
        Path to the Excel file containing skills data
        
    Returns:
    --------
    pd.DataFrame
        Cleaned skills data with Country, Industry, Skill, and Skill Rank columns
    """
    try:
        # Load and clean skills data
        data_corrected = pd.read_excel(skills_file, sheet_name="2A - SGP Ctry Ind", header=3)
        
        data_cleaned = data_corrected.rename(columns={
            'Unnamed: 2': 'Country',
            'Unnamed: 3': 'Industry',
            'Unnamed: 4': 'Skill',
            'Unnamed: 5': 'Skill Rank'
        })
        
        data_cleaned = data_cleaned.dropna(subset=['Country', 'Industry', 'Skill', 'Skill Rank'])
        data_cleaned = data_cleaned.iloc[4:]
        data_cleaned = data_cleaned.drop(columns=['Unnamed: 0', 'Unnamed: 1'])
        
        # Convert Skill Rank to numeric
        data_cleaned['Skill Rank'] = pd.to_numeric(data_cleaned['Skill Rank'], errors='coerce')
        data_cleaned = data_cleaned.dropna(subset=['Skill Rank'])
        
        return data_cleaned
    except Exception as e:
        print(f"Error during skills data cleaning: {str(e)}")
        return None


def clean_skill_penetration_data(excel_file, sheet_name="3A - SPP Ctry"):
    """
    Clean and prepare skill penetration data from Excel file.
    
    Parameters:
    -----------
    excel_file : str
        Path to the Excel file containing skill penetration data
    sheet_name : str
        Name of the sheet to read (default: "3A - SPP Ctry")
        
    Returns:
    --------
    pd.DataFrame
        Cleaned skill penetration data with Country, Skill, Average, Global, and Relative columns
    """
    try:
        data = pd.read_excel(excel_file, sheet_name=sheet_name, header=5)

        required_columns = ['Country', 'Skill', 'Average', 'Global', 'Relative']
        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            print("Missing columns:", missing)
            print("Available:", data.columns.tolist())
            return None

        data = data.dropna(subset=required_columns)
        data['Country'] = data['Country'].str.strip()

        return data
    except Exception as e:
        print(f"Error during skill penetration data cleaning: {str(e)}")
        return None


def clean_skill_penetration_by_industry_data(excel_file, sheet_name="3B - SPP Ctry Ind"):
    """
    Clean and prepare skill penetration by industry data from Excel file.
    
    Parameters:
    -----------
    excel_file : str
        Path to the Excel file containing skill penetration by industry data
    sheet_name : str
        Name of the sheet to read (default: "3B - SPP Ctry Ind")
        
    Returns:
    --------
    pd.DataFrame
        Cleaned skill penetration data with Country, Industry, Skill, Average, Global, and Relative columns
    """
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=5)

        required = ["Country", "Industry", "Skill", "Average", "Global", "Relative"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print("Missing:", missing)
            print("Available:", df.columns.tolist())
            return None

        df = df.dropna(subset=required)
        df["Country"] = df["Country"].str.strip()
        df["Industry"] = df["Industry"].str.strip()
        df["Skill"] = df["Skill"].str.strip()

        return df
    except Exception as e:
        print(f"Error during skill penetration by industry data cleaning: {str(e)}")
        return None


def export_skills_metrics(skills_file, lhr_file, output_path='industry_skills_metrics.csv'):
    """
    Export skills rankings and metrics for each industry category.
    
    Parameters:
    -----------
    skills_file : str
        Path to the Excel file containing skills data
    lhr_file : str
        Path to the Excel file containing LHR data (to get industry categories)
    output_path : str
        Path where the CSV file should be saved
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with skills metrics by category and industry
    """
    import os
    
    try:
        # Get industry categories from LHR data
        industry_dict = create_industry_performance_dict(lhr_file)
        
        # Clean skills data
        skills_data = clean_skills_data(skills_file)
        
        if skills_data is None or industry_dict is None:
            return None
        
        # Filter for country
        country = os.environ.get('COUNTRY', 'Algeria')
        bd_skills = skills_data[skills_data['Country'] == country].copy()
        
        # Prepare data for export
        export_data = []
        
        categories = [
            ('Leading Industries', industry_dict['leading_industries']),
            ('Growing Industries', industry_dict['growing_industries']),
            ('Transitioning Industries', industry_dict['transitioning_industries']),
            ('Emerging Industries', industry_dict['emerging_industries'])
        ]
        
        for category_name, industries in categories:
            for industry in industries.keys():
                industry_skills = bd_skills[bd_skills['Industry'] == industry].copy()
                
                if not industry_skills.empty:
                    # Get top 5 skills
                    top_skills = industry_skills.nsmallest(5, 'Skill Rank')
                    
                    # Add each skill to export data
                    for _, skill_row in top_skills.iterrows():
                        export_data.append({
                            'Category': category_name,
                            'Industry': industry,
                            'Skill': skill_row['Skill'],
                            'Skill_Rank': skill_row['Skill Rank']
                        })
        
        # Convert to DataFrame and export
        metrics_df = pd.DataFrame(export_data)
        metrics_df.to_csv(output_path, index=False)
        print(f"Skills metrics exported to {output_path}")
        
        return metrics_df
        
    except Exception as e:
        print(f"Error exporting skills metrics: {str(e)}")
        return None

