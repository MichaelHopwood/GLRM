import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class StockBlockLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, stack_cnt=0):
        super(StockBlockLayer, self).__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer
        self.weight = nn.Parameter(
            torch.Tensor(1, 3 + 1, 1, self.time_step * self.multi,
                         self.multi * self.time_step))  # [K+1, 1, in_c, out_c]
        nn.init.xavier_normal_(self.weight)
        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)
        if self.stack_cnt == 0:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)
        self.relu = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi
        for i in range(3):
            if i == 0:
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
            elif i == 1:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
            else:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))

    def spe_seq_cell(self, input):
        print('\tInto SPE SEQ CELL', input.size())
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step)
        print('\tview input', input.size())
        ffted = torch.rfft(input, 1, onesided=False)
        print('\tffted', ffted.size())
        real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        print('\treal', real.size())
        img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        print('\timg', img.size())
        print('\tStarting GLUs')
        for i in range(3):
            print('\t\t iteration #', i)
            real = self.GLUs[i * 2](real)
            print('\t\t real', real.size())
            img = self.GLUs[2 * i + 1](img)
            print('\t\t img', img.size())
        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        print('\treal and unsqueeze real', real.size(), real.unsqueeze(-1).size())
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        print('\timg and unsqueeze img', img.size(), img.unsqueeze(-1).size())
        time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
        print('\tTimestep as inner = cat(real unsqueeze, img unsqueeze, dim=-1)', time_step_as_inner.size())
        iffted = torch.irfft(time_step_as_inner, 1, onesided=False)
        print('\tOut iffted', iffted.size())
        return iffted

    def forward(self, x, mul_L):
        print('In stock block')
        print('\tx', x.size())
        print('\tmul_L', mul_L.size())
        mul_L = mul_L.unsqueeze(1)
        print('\tunsqueeze 1 on mulL', mul_L.size())
        x = x.unsqueeze(1)
        print('\tunsqueeze 1 on x', x.size())
        gfted = torch.matmul(mul_L, x)
        print('\tgfted=mul_L*x', gfted.size())
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)
        print('\tgconv_input = output iffted unsqueeze 2: ', gconv_input.size())
        igfted = torch.matmul(gconv_input, self.weight)
        print('\tigfted = gconv_input * weight', igfted.size, gconv_input.size, self.weight.size())
        igfted = torch.sum(igfted, dim=1)
        print('\tigfted summed on dim 1:', igfted.size())
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        print('\tsigmoid( forecast squeezed ):', forecast_source.size())
        forecast = self.forecast_result(forecast_source)
        print('\tforecast_result: ', forecast.size())
        if self.stack_cnt == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1)
            print('\tbackcast linear layer on x and squeezed', backcast_short.size())
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
            print('\tsigmoid ( backcast linear layer on igfted MINUS backcast_short )', backcast_source.size())
            print('\tout stock block: backcast', backcast_source.size())
        else:
            backcast_source = None
            print('\tout stock block: backcast', backcast_source)
        print('\tout stock block: forecast', forecast.size())
        
        return forecast, backcast_source


class Model(nn.Module):
    def __init__(self, units, stack_cnt, time_step, multi_layer, horizon=1, dropout_rate=0.5, leaky_rate=0.2,
                 device='cpu'):
        super(Model, self).__init__()
        self.unit = units
        self.stack_cnt = stack_cnt
        self.unit = units
        self.alpha = leaky_rate
        self.time_step = time_step
        self.horizon = horizon
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.GRU = nn.GRU(self.time_step, self.unit)
        self.multi_layer = multi_layer
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [StockBlockLayer(self.time_step, self.unit, self.multi_layer, stack_cnt=i) for i in range(self.stack_cnt)])
        self.fc = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), self.horizon),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.to(device)

    def get_laplacian(self, graph, normalize):
        """
        return the laplacian of the graph.
        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        print('laplacian into chebychev polynomial:', laplacian.size())
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        print('unsqueeze')
        first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian

    def latent_correlation_layer(self, x):
        print('reformat:',x.permute(2, 0, 1).contiguous().size())
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        print('After GRU', input.size())
        input = input.permute(1, 0, 2).contiguous()
        print('After reformat', input.size())
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)
        print('Attention mean\'d on dim 0', attention.size())
        degree = torch.sum(attention, dim=1)
        print('Degree = attention sum on dim 1', degree.size())
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)
        print('attention refactor', attention.size())
        degree_l = torch.diag(degree)
        print('degree_L', degree_l.size())
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        print('diagonal_degree_hat', diagonal_degree_hat.size())
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))
        print('laplacian = diagonal_degree_hat * ((degree_l - attention) * diagonal_degree_hat)', laplacian.size())
        mul_L = self.cheb_polynomial(laplacian)
        print('mul_L size', mul_L.size())
        print('final attention size', attention.size())
        return mul_L, attention

    def self_graph_attention(self, input):
        print('into graph attention: ', input.size())
        input = input.permute(0, 2, 1).contiguous()
        print('reformat', input.size())
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        print('key=Input*WeightKey size', key.size(), '=', input.size, self.weight_key.size())
        query = torch.matmul(input, self.weight_query)
        print('query=input*weightQuery size', query.size(), '=', input.size(), self.weight_query.size())
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        print('define data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)', data.size(), key.repeat(1, 1, N).view(bat, N * N, 1).size(), query.repeat(1, N, 1).size())
        data = data.squeeze(2)
        print('squeeze data:', data.size())
        data = data.view(bat, N, -1)
        print('view data', data.size())
        data = self.leakyrelu(data)
        print('pass leakyrelu', data.size())
        attention = F.softmax(data, dim=2)
        print('compute attention', attention.size())
        attention = self.dropout(attention)
        print("dropout on attention: ", attention.size())
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def forward(self, x):
        print("Input size:", x.size())
        mul_L, attention = self.latent_correlation_layer(x)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        print("X squeezed size:", X.size())
        result = []
        for stack_i in range(self.stack_cnt):
            print('starting stock block #', stack_i)
            forecast, X = self.stock_block[stack_i](X, mul_L)
            result.append(forecast)
        forecast = result[0] + result[1]
        print('redefine forecast as first and last result', forecast.size())
        forecast = self.fc(forecast)
        print('MLP on forecast', forecast.size())
        if forecast.size()[-1] == 1:
            print('A')
            return forecast.unsqueeze(1).squeeze(-1), attention
        else:
            print('B')
            return forecast.permute(0, 2, 1).contiguous(), attention
