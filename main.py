import torch
import torch.nn as nn
import torch.optim as optim #optimierungsalgorithmen
import numpy as np

# Sequenz wird definiert --> array besteht aus 5 zeitschritten (jeder zeitschritt hat einen vektor mit 2 werten)
seq = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]])

# model wird definiert (neuronale netz)
class LSTMNet(nn.Module): # klasse erbt von nn.module
    def __init__(self):
        super(LSTMNet, self).__init__()
        # gewichtsmatritzen und bias vektoren
        self.U_f = nn.Parameter(torch.Tensor(2, 2))
        self.V_f = nn.Parameter(torch.Tensor(2, 2))
        self.b_f = nn.Parameter(torch.Tensor(1, 2))
        self.U_i = nn.Parameter(torch.Tensor(2, 2))
        self.V_i = nn.Parameter(torch.Tensor(2, 2))
        self.b_i = nn.Parameter(torch.Tensor(1, 2))
        self.U_o = nn.Parameter(torch.Tensor(2, 2))
        self.V_o = nn.Parameter(torch.Tensor(2, 2))
        self.b_o = nn.Parameter(torch.Tensor(1, 2))
        self.U_g = nn.Parameter(torch.Tensor(2, 2))
        self.V_g = nn.Parameter(torch.Tensor(2, 2))
        self.b_g = nn.Parameter(torch.Tensor(1, 2))
        self.init_weights() # biases und gewichte werden zufällig initialisiert

    def init_weights(self): # methode die die biases und gewichte zufällig initialisiert
        for param in self.parameters(): # iterieren durch alle parameter des modells
            nn.init.uniform_(param, -0.1, 0.1) # jeder parameter wird zufällig initialisiert mit einer random zahl zwischen -0.1 und 0.1

    def forward(self, input, hx=None): # vorwärtsmethode wurde definiert die den input sequenziell durch das lstm netz durchlaufen lässt
        h_t, c_t = hx or (torch.zeros(1, 2), torch.zeros(1, 2)) # hidden state und cell state mit nullen werden initialisert ( falls keine startwerte angegeben worden sind)???
        output_seq = [] # leere liste, dient dazu das ergebniss der funktion zu speichern
        for x_t in input:
            f_t = torch.sigmoid(torch.matmul(x_t, self.U_f) + torch.matmul(h_t, self.V_f) + self.b_f)
            i_t = torch.sigmoid(torch.matmul(x_t, self.U_i) + torch.matmul(h_t, self.V_i) + self.b_i)
            o_t = torch.sigmoid(torch.matmul(x_t, self.U_o) + torch.matmul(h_t, self.V_o) + self.b_o)
            g_t = torch.tanh(torch.matmul(x_t, self.U_g) + torch.matmul(h_t, self.V_g) + self.b_g)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            output_seq.append(h_t)
        return torch.stack(output_seq, dim=1)

# Define the training function
def train(net, seq):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(20): # funktion, iteriert über 20 epochen --> vorwärts und rückwärtspropagation
        optimizer.zero_grad() #aktualisiert parameter des netzes um fehler zu minimieren
        input_seq = torch.Tensor(seq).unsqueeze(0)
        output_seq = net(input_seq)
        target_seq = torch.Tensor(seq).unsqueeze(0) # ausgabe des netzes wird wird mit ziel sequenz verglichen,
        loss = criterion(output_seq, target_seq) # loose wird berechnet
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Train the model and print the final output sequence
net = LSTMNet()
train(net, seq) # train funktion --> trainiert das modell
input_seq = torch.Tensor(seq).unsqueeze(0) #Die Eingabe-Sequenz "seq" wird in eine PyTorch Tensor-Variable
# umgewandelt und eine zusätzliche Dimension wird hinzugefügt, um das Batch-Format für die LSTM-Eingabe zu erstellen.
output_seq = net(input_seq) # Die Eingabe-Sequenz wird dem trainierten LSTM-Netzwerk "net" übergeben und das Ausgabe-Sequenz wird generiert.
print(f'Input Sequence: \n{seq}') # input sequenz wird ausgegeben
print(f'Output Sequence: \n{output_seq.detach().numpy()[0]}') # erstes element des arrays wird ausgegeben

