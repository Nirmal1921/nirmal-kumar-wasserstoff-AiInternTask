def map_data(identified_objects, master_image_id="image_001"):
    mapped_data = []
    for obj in identified_objects:
        if isinstance(obj, dict):  # Ensure obj is a dictionary
            mapped_data.append({
                'object_id': obj.get('label', 'N/A'),
                'object_name': obj.get('label', 'N/A'),
                'object_description': obj.get('description', 'N/A'),
                'extracted_text': obj.get('extracted_text', 'N/A'),
                'summary': obj.get('summary', 'N/A')
            })
    return {
        'master_image_id': master_image_id,
        'data': mapped_data
    }
