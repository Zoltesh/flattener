import pandas as pd
import json

def flatten_nested_json(df, parent_id_col='id', prefix='', source_file=''):
    """
    Recursively flatten nested JSON arrays in a dataframe and save as separate parquet files.
    
    Args:
        df: Input DataFrame
        parent_id_col: Column name containing the parent ID
        prefix: Prefix for output parquet files
        source_file: Original source file name for prefix generation
    """
    # Generate base prefix from source file if not provided
    if not prefix:
        prefix = source_file.replace('.parquet', '') + '_'
    
    # Keep track of created files to avoid duplicates
    processed_columns = set()
    
    for column in df.columns:
        # Skip the ID column and already processed columns
        if column == parent_id_col or column in processed_columns:
            continue
            
        # Check if the column contains JSON arrays
        if df[column].dtype == 'object':
            # Get all non-null values
            valid_values = df[column].dropna()
            if valid_values.empty:
                continue
                
            try:
                # Try to parse each value and collect valid JSON arrays/objects
                nested_data = []
                parent_ids = []
                
                for idx, value in valid_values.items():
                    try:
                        parsed = json.loads(value) if isinstance(value, str) else value
                        if isinstance(parsed, (list, dict)):
                            # Handle both single objects and lists of objects
                            if isinstance(parsed, dict):
                                parsed = [parsed]
                            nested_data.extend(parsed)
                            parent_ids.extend([df.loc[idx, parent_id_col]] * len(parsed))
                    except (json.JSONDecodeError, TypeError):
                        continue
                
                if nested_data:
                    # Create DataFrame from nested data
                    nested_df = pd.json_normalize(nested_data)
                    
                    if not nested_df.empty:
                        # Add reference to parent ID
                        nested_df[f'parent_{parent_id_col}'] = parent_ids
                        
                        # Save as parquet file
                        output_file = f"{prefix}{column}.parquet"
                        nested_df.to_parquet(output_file, index=False)
                        processed_columns.add(column)
                        
                        # Recursively process nested columns
                        flatten_nested_json(
                            nested_df, 
                            parent_id_col=f'parent_{parent_id_col}',
                            prefix=f"{prefix}{column}_",
                            source_file=source_file
                        )
            except Exception as e:
                print(f"Error processing column {column}: {str(e)}")
                continue

    # Create cleaned version of original dataframe without nested columns
    clean_df = df.drop(columns=processed_columns)
    return clean_df

def main():
    # Read the input parquet file
    input_file = "data/orders/orders.parquet"
    source_filename = input_file.split('/')[-1]
    
    try:
        df = pd.read_parquet(input_file)
        print(f"Processing {input_file}")
        print(f"Found columns: {', '.join(df.columns)}")
        
        # Flatten nested JSON and save new parquet files
        clean_df = flatten_nested_json(df, source_file=source_filename)
        
        # Save the flattened main file
        output_file = f"{source_filename.replace('.parquet', '')}_flattened.parquet"
        clean_df.to_parquet(output_file, index=False)
        print(f"Successfully created {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
