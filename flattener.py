import pandas as pd
import json
import os

def identify_id_column(df):
    """Identify the likely ID column from the DataFrame."""
    # Common ID column names
    id_patterns = ['id', 'key', 'identifier', 'uuid']
    
    for pattern in id_patterns:
        matches = [col for col in df.columns if pattern.lower() in col.lower()]
        if matches:
            # Prefer exact matches first
            exact_matches = [col for col in matches if col.lower() == pattern]
            return exact_matches[0] if exact_matches else matches[0]
    
    # If no ID column found, use the first column
    return df.columns[0]

def flatten_nested_json(df, parent_id_col='id', prefix='', source_file='', temp_dir='temp'):
    """
    Recursively flatten nested JSON arrays in a dataframe and save as separate parquet files.
    
    Args:
        df: Input DataFrame
        parent_id_col: Column name containing the parent ID
        prefix: Prefix for output parquet files
        source_file: Original source file name for prefix generation
        temp_dir: Directory for temporary files
    """
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate base prefix from source file if not provided
    if not prefix:
        prefix = source_file.replace('.parquet', '') + '_'
    
    # Keep track of processed columns and created files
    processed_columns = set()
    created_files = []
    
    for column in df.columns:
        # Skip the ID column and already processed columns
        if column == parent_id_col or column in processed_columns:
            continue
            
        if df[column].dtype == 'object':
            valid_values = df[column].dropna()
            if valid_values.empty:
                continue
                
            try:
                nested_data = []
                parent_ids = []
                
                for idx, value in valid_values.items():
                    try:
                        parsed = json.loads(value) if isinstance(value, str) else value
                        if isinstance(parsed, (list, dict)):
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
                        # Identify ID column in nested data
                        nested_id_col = identify_id_column(nested_df)
                        
                        # Add reference to parent ID
                        nested_df[f'parent_{parent_id_col}'] = parent_ids
                        
                        # Save temporary file
                        temp_file = os.path.join(temp_dir, f"{prefix}{column}.parquet")
                        nested_df.to_parquet(temp_file, index=False)
                        
                        # Recursively process nested columns
                        sub_clean_df, sub_files = flatten_nested_json(
                            nested_df,
                            parent_id_col=nested_id_col,
                            prefix=f"{prefix}{column}_",
                            source_file=source_file,
                            temp_dir=temp_dir
                        )
                        
                        # If nested data was further flattened, use the flattened version
                        if sub_files:
                            created_files.extend(sub_files)
                            # Save flattened version
                            output_file = f"{prefix}{column}_flattened.parquet"
                            sub_clean_df.to_parquet(output_file, index=False)
                            created_files.append(output_file)
                            # Remove temporary file
                            os.remove(temp_file)
                        else:
                            # If no further flattening needed, rename temp file to final
                            output_file = f"{prefix}{column}.parquet"
                            os.rename(temp_file, output_file)
                            created_files.append(output_file)
                        
                        processed_columns.add(column)
                        
            except Exception as e:
                print(f"Error processing column {column}: {str(e)}")
                continue

    # Create cleaned version of original dataframe without nested columns
    clean_df = df.drop(columns=processed_columns)
    return clean_df, created_files

def main():
    # Read the input parquet file
    input_file = "data/inbound_shipments/inbound_shipments.parquet"
    source_filename = input_file.split('/')[-1]
    temp_dir = "temp_parquet_files"
    
    try:
        df = pd.read_parquet(input_file)
        print(f"Processing {input_file}")
        print(f"Found columns: {', '.join(df.columns)}")
        
        # Flatten nested JSON and save new parquet files
        clean_df, created_files = flatten_nested_json(
            df, 
            source_file=source_filename,
            temp_dir=temp_dir
        )
        
        # Save the flattened main file
        output_file = f"{source_filename.replace('.parquet', '')}_flattened.parquet"
        clean_df.to_parquet(output_file, index=False)
        print(f"Successfully created {output_file}")
        print("Created additional files:")
        for file in created_files:
            print(f"- {file}")
        
        # Clean up temp directory if empty
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
