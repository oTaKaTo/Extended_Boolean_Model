"""
Helper script to extract JSON ranking from Gemini output files
"""

import json
import re
import os
import glob


def extract_json_from_gemini_file(filepath):
    """
    Extract JSON ranking from Gemini output text file
    
    Args:
        filepath: Path to the gemini output text file
        
    Returns:
        Dictionary with ranking data or None if not found
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find JSON object in the response
        # Look for pattern: { "ranking": [ ... ] }
        json_match = re.search(r'\{[\s\S]*"ranking"[\s\S]*\][\s\S]*\}', content)
        
        if json_match:
            json_str = json_match.group(0)
            # Try to parse it
            data = json.loads(json_str)
            return data
        else:
            print(f"No JSON ranking found in {filepath}")
            return None
            
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def list_gemini_files():
    """List all Gemini output files in current directory"""
    files = glob.glob("gemini_*.txt")
    files.sort(reverse=True)  # Most recent first
    return files


def main():
    print("=" * 70)
    print("GEMINI RANKING EXTRACTOR")
    print("=" * 70)
    print()
    
    # List available files
    files = list_gemini_files()
    
    if not files:
        print("No Gemini output files found (gemini_*.txt)")
        return
    
    print(f"Found {len(files)} Gemini output file(s):")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")
    print()
    
    # Get user choice
    try:
        choice = input(f"Select file (1-{len(files)}) or press Enter for most recent: ").strip()
        
        if choice == "":
            selected_file = files[0]
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                selected_file = files[idx]
            else:
                print("Invalid selection")
                return
        
        print(f"\nProcessing: {selected_file}")
        print("-" * 70)
        
        # Extract JSON
        data = extract_json_from_gemini_file(selected_file)
        
        if data and 'ranking' in data:
            ranking = data['ranking']
            print(f"\n✓ Found {len(ranking)} ranked documents")
            print()
            
            # Display top 5
            print("Top 5 Results:")
            print("-" * 70)
            for item in ranking[:5]:
                print(f"  Rank {item['rank']}: D{item['document']} - Similarity: {item['similarity']}")
            print()
            
            # Pretty print full JSON
            print("Full JSON Output (copy this to Streamlit GUI):")
            print("=" * 70)
            print(json.dumps(data, indent=2))
            print("=" * 70)
            
            # Save to separate file
            output_file = selected_file.replace('.txt', '_ranking.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"\n✓ Saved to: {output_file}")
            
        else:
            print("✗ Could not extract ranking data")
    
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
    except ValueError:
        print("Invalid input")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
