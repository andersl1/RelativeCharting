def downsample_data(data, n):
    """
    Downsamples a pandas Series by selecting every nth element, 
    but includes global min/max within each interval if they 
    are the global min/max seen thus far.

    Args:
        data: A pandas Series.
        n: The interval for downsampling.

    Returns:
        A pandas Series with downsampled data.
    """

    if n <= 0:
        raise ValueError("n must be a positive integer.")

    if n >= len(data):
        return data  # Return original if n is too large.

    downsampled_data = []
    downsampled_indices = []

    min_val = data.iloc[0]
    max_val = data.iloc[0]

    min_val_interval = data.iloc[0]
    max_val_interval = data.iloc[0]

    for i in range(len(data)):
        if data.iloc[i] < min_val:
            min_val = data.iloc[i]
            min_val_interval = min_val
        if data.iloc[i] > max_val:
            max_val = data.iloc[i]
            max_val_interval = max_val

        if i % n == 0:
            if max_val_interval == max_val:
                downsampled_data.append(max_val_interval)
                downsampled_indices.append(data.index[i])
            elif min_val_interval == min_val:
                downsampled_data.append(min_val_interval)
                downsampled_indices.append(data.index[i])
            else:
                downsampled_data.append(data.iloc[i])
                downsampled_indices.append(data.index[i])

            if i < (len(data) - 1):
                min_val_interval = data.iloc[i + 1]
                max_val_interval = data.iloc[i + 1]

    return pd.Series(downsampled_data, index=downsampled_indices)


def construct_smooth(data, smooth_value):
    
    print(":smooth_value:", smooth_value)
    slider_values = data.get('sliderValues', {})

    cache_id = data.get('cache_id')

    if not cache_id:
        return jsonify({'error': 'Missing cache ID'}), 400

    cache_entry = DATA_CACHE.get(cache_id)
    if not cache_entry:
        return jsonify({'error': 'Invalid or expired cache ID'}), 400

    cached_data = cache_entry['cached_data']
    print(f"Processing cache ID: {cache_id}")

    # Validate data structure
    if not all('dates' in d and 'investment' in d for d in cached_data.values()):
        raise ValueError("Invalid cached data structure")

    # Create a DataFrame with all dates
    all_dates = sorted(set().union(*[d['dates'] for d in cached_data.values()]))
    df = pd.DataFrame(index=all_dates)
    total_investment = 0

    # Add weighted investments
    for symbol, weight_pct in slider_values.items():
        if symbol in cached_data:
            weight = weight_pct / 100.0
            # print("GETTING HERE: weight is:", weight, "for symbol:", symbol)
            symbol_data = cached_data[symbol]
            temp_df = pd.DataFrame({
                'investment': symbol_data['investment'],
                'date': symbol_data['dates'],
                'weight': symbol_data['weight']
            }).set_index('date')
            df[symbol] = temp_df['investment'] * weight
            total_investment += symbol_data['weight'] * weight


    # Sum across all symbols and handle NaN
    combined = df.sum(axis=1).fillna(0)

    # Downsample the data
    try:
        downsampled = downsample_data(combined, smooth_value)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    return downsampled, total_investment
