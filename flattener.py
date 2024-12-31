import pandas as pd
import json
import os
from typing import Dict, List, Any

def flatten_parquet(input_parquet: str, parent_name: str, identifier: str) -> None:
    """
    Flattens a nested parquet file into multiple tables with relational structure.
    
    Parameters:
    - input_parquet: Path to the input parquet file.
    - parent_name: Name to use as the base for output file names (e.g., 'orders').
    - identifier: The column to use as the unique identifier for the top-level records.
    """
    # Load the original Parquet file
    try:
        df = pd.read_parquet(input_parquet)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    if identifier not in df.columns:
        print(f"Error: Identifier column '{identifier}' not found in the parquet file")
        return

    # Initialize a dictionary to store flattened data
    flattened_data: Dict[str, List[Dict[str, Any]]] = {}
    # Track columns that contain nested data
    nested_columns = set()

    def process_nested(obj: Any, parent_keys: List[str], parent_id: Any, original_id: Any = None) -> None:
        """
        Recursively processes a nested object.
        
        Parameters:
        - obj: The current object (could be a dict, list, or scalar).
        - parent_keys: A list of keys representing the current hierarchy.
        - parent_id: The identifier value from the parent object.
        - original_id: The original top-level order ID to maintain throughout
        """
        # Set original_id on first call
        if original_id is None:
            original_id = parent_id

        # Handle JSON strings
        if isinstance(obj, str):
            try:
                parsed_obj = json.loads(obj)
                if isinstance(parsed_obj, (dict, list)):
                    obj = parsed_obj
            except json.JSONDecodeError:
                pass

        if isinstance(obj, dict):
            # Create a table for this dictionary
            table_name = "_".join(parent_keys)
            if table_name not in flattened_data:
                flattened_data[table_name] = []
            
            # Add all scalar values to this table
            row_data = {f"{parent_name}_{identifier}": original_id}  # Use original_id here
            has_scalar_values = False
            nested_items = {}
            
            # First pass: separate scalar and nested values
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    nested_items[key] = value
                else:
                    row_data[key] = value
                    has_scalar_values = True
            
            if has_scalar_values:
                # Generate a unique ID for this record
                record_id = str(obj.get('id', '')) or str(obj.get(f'{table_name}_id', '')) or str(len(flattened_data[table_name]))
                row_data['id'] = record_id
                flattened_data[table_name].append(row_data)
                
                # Process nested items with the current record's ID
                for key, value in nested_items.items():
                    process_nested(value, parent_keys + [key], record_id, original_id)
            
            elif nested_items:  # Process nested items even if no scalar values
                record_id = parent_id
                for key, value in nested_items.items():
                    process_nested(value, parent_keys + [key], record_id, original_id)
                
        elif isinstance(obj, list):
            # Handle list of objects
            table_name = "_".join(parent_keys)
            if table_name not in flattened_data:
                flattened_data[table_name] = []
            
            for i, item in enumerate(obj):
                if isinstance(item, dict):
                    # Add parent reference and index
                    row_data = {
                        f"{parent_name}_{identifier}": original_id,  # Use original_id here
                        "index": i
                    }
                    
                    # Separate scalar and nested values
                    nested_items = {}
                    for key, value in item.items():
                        if isinstance(value, (dict, list)):
                            nested_items[key] = value
                        else:
                            row_data[key] = value
                    
                    # Store the current row_data
                    flattened_data[table_name].append(row_data)
                    
                    # Process nested structures
                    item_id = str(item.get('id', '')) or str(item.get(f'{table_name}_id', '')) or f"{parent_id}_{i}"
                    for key, value in nested_items.items():
                        process_nested(value, parent_keys + [key], item_id, original_id)
                else:
                    # Handle scalar values in lists
                    flattened_data[table_name].append({
                        f"{parent_name}_{identifier}": original_id,  # Use original_id here
                        "index": i,
                        "value": item
                    })

        # Track nested columns (keep this part)
        if len(parent_keys) == 1 and isinstance(obj, (dict, list)):
            nested_columns.add(parent_keys[0])

    # Process each row in the input dataframe
    for idx, row in df.iterrows():
        parent_id = row[identifier]
        try:
            for column_name, value in row.items():
                # Handle Series objects directly
                if isinstance(value, pd.Series):
                    value = value.iloc[0]
                
                # Process the value without null checks
                process_nested(value, [column_name], parent_id)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    # Create output directory in the same location as input file
    input_dir = os.path.dirname(input_parquet)
    output_dir = os.path.join(input_dir, f"{parent_name}_flattened")
    os.makedirs(output_dir, exist_ok=True)

    # First save all the nested structure files
    for table_name, rows in flattened_data.items():
        if rows:  # Skip if no data
            try:
                table_df = pd.DataFrame(rows)
                output_path = os.path.join(output_dir, f"{parent_name}_{table_name}.parquet")
                table_df.to_parquet(output_path, index=False)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error saving table {table_name}: {e}")

    # Finally, create and save the flattened main file
    try:
        # Remove nested columns from the main DataFrame
        flattened_df = df.drop(columns=list(nested_columns))
        output_path = os.path.join(output_dir, f"{parent_name}_flattened.parquet")
        flattened_df.to_parquet(output_path, index=False)
        print(f"Saved flattened main file: {output_path}")
    except Exception as e:
        print(f"Error saving flattened main file: {e}")

if __name__ == "__main__":
    # Example usage
    flatten_parquet(
        input_parquet="data/orders/orders.parquet",
        parent_name="orders",
        identifier="id"
    )
