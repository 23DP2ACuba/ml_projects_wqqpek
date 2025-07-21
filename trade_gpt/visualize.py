def visualize_predictions(start_idx=0, feature_idx=3, feature_name="Close", unscale=True):
    """
    Visualize model predictions vs actual values for a selected feature in the validation set.
    """
    model.eval()
    with torch.no_grad():
        input_seq = val_data[start_idx : start_idx + block_size].unsqueeze(0).to(device)
        generated = model.generate(input_seq, max_new_tokens=block_size)
        predicted = generated[:, -block_size:, :].squeeze(0).cpu().numpy()

    actual = val_data[start_idx + block_size : start_idx + 2 * block_size].cpu().numpy()

    # Extract the feature to compare
    pred_series = predicted[:, feature_idx]
    actual_series = actual[:, feature_idx]

    if unscale:
        full = np.concatenate([predicted, actual], axis=0)
        full_unscaled = scaler.inverse_transform(full)
        pred_series = full_unscaled[:block_size, feature_idx]
        actual_series = full_unscaled[block_size:, feature_idx]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(block_size), pred_series, label="Predicted", color="blue", linewidth=2)
    plt.plot(range(block_size), actual_series, label="Actual", color="orange", linewidth=2)
    plt.title(f"{feature_name} Prediction ({SYMBOL})")
    plt.xlabel("Days Ahead")
    plt.ylabel(f"{feature_name} ({'Unscaled' if unscale else 'Scaled'})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

visualize_predictions()
