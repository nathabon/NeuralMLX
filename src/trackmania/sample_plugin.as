Net::Socket@ sock = null;
Net::Socket@ clientSock = null;

enum MessageType {
    SCRunStepSync = 0,
    CSetSpeed,
    CRewindToState,
    CSetInputState,
    CShutdown
}

const string HOST = "127.0.0.1";
const uint16 PORT = 8477;
const uint RESPONSE_TIMEOUT = 2000;

void WaitForResponse(MessageType type)
{
    auto now = Time::Now;

    while (true) {
        auto receivedType = HandleMessage();
        if (receivedType == type) {
            break;
        }

        if (receivedType == MessageType::CShutdown) {
            break;
        }

        if (receivedType == -1 && Time::Now - now > RESPONSE_TIMEOUT) {
            log("Client disconnected due to timeout (" + RESPONSE_TIMEOUT + "ms)");
            @clientSock = null;
            break;
        }
    }
}

int HandleMessage()
{
    if (clientSock.Available == 0) {
        return -1;
    }

    auto type = clientSock.ReadInt32();
    switch(type) {
        case MessageType::SCRunStepSync: {
            break;
        }

        case MessageType::CSetSpeed: {
            auto@ simManager = GetSimulationManager();
            simManager.SetSpeed(clientSock.ReadFloat());
            break;
        }

        case MessageType::CRewindToState: {
            auto stateLength = clientSock.ReadInt32();
            auto stateData = clientSock.ReadBytes(stateLength);
            auto@ simManager = GetSimulationManager();
            if (simManager.InRace) {
                SimulationState state(stateData);
                simManager.RewindToState(state);
            }

            break;            
        }


        case MessageType::CSetInputState: {
            int8 up   = clientSock.ReadInt8();
            int8 down = clientSock.ReadInt8();
            int steer = clientSock.ReadInt32();

            auto@ simManager = GetSimulationManager();
            if (simManager.InRace) {
                log("Set input state");
                simManager.SetInputState(InputType::Up, up);
                simManager.SetInputState(InputType::Down, down);
                if (steer != 0x7FFFFFFF) {
                    simManager.SetInputState(InputType::Steer, steer);
                }
            }

            break;
        }

        case MessageType::CShutdown: {
            log("Client disconnected");
            @clientSock = null;
            break;
        }
    }

    return type;
}

void OnRunStep(SimulationManager@ simManager)
{
    if (@clientSock is null) {
        return;
    }

    auto@ state = simManager.SaveState();
    auto@ data = state.ToArray();

    clientSock.Write(MessageType::SCRunStepSync);
    clientSock.Write(data.Length);
    clientSock.Write(data);
    WaitForResponse(MessageType::SCRunStepSync);
}

void Main()
{
    if (@sock is null) {
        @sock = Net::Socket();
        sock.Listen(HOST, PORT);
    }
}

void Render()
{
    auto @newSock = sock.Accept(0);
    if (@newSock !is null) {
        @clientSock = @newSock;
        log("Client connected (IP: " + clientSock.RemoteIP + ")");
    }
}

PluginInfo@ GetPluginInfo()
{
    PluginInfo info;
    info.Author = "donadigo";
    info.Name = "Test";
    info.Description = "Sockets example";
    info.Version = "1.0.0";
    return info;
}