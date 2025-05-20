class TrainPipeline:
    def __init__(self, model, data_loader, optimizer, loss_fn):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, epochs):
        for epoch in range(epochs):
            for batch in self.data_loader:
                # Forward pass
                predictions = self.model(batch["input"])
                loss = self.loss_fn(predictions, batch["target"])

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
