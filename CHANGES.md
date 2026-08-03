---
[[Baseline]]
# M2T2 — Tasks de Integração (server ZMQ + bridge ROS 2)

Contexto: container `m2t2:latest` já funcional (build via `docker/Dockerfile`, CUDA 11.7.1-devel, `pointnet2_ops` compilando OK, `demo.py` validado com `object_00 has 7 grasps`). Falta: transformar o `demo.py` script único num servidor ZMQ de longa duração, e ligar isso ao pipeline ROS 2 existente (`grasp_baseline`, já usado com GraspGen).

Escopo fechado: **apenas modo grasping** (pick). Modo placing fora do escopo.

---

## Parte 1 — Repositório M2T2 (`M2T2/`)
path: 
```bash
(m2t2_env) nexus@nexus-PC:~/Desktop/M2T2 (master)$ ls
CHANGES.md     demo_rlbench.py  LICENSE        m2t2_env       REPORT.md         setup.py
client-server  docker           logs.txt       outputs        requirements.txt  train.py
config.yaml    figures          m2t2           pointnet2_ops  rlbench.yaml      weights
demo.py        justfile         m2t2.egg-info  README.md      sample_data

```

### 1.1 Confirmar valores de config antes de codar
- [x] Ler `config.yaml` (raiz do repo) e anotar os valores default de: `cfg.data.num_points`, `cfg.eval.mask_thresh`, `cfg.eval.object_thresh`, `cfg.eval.world_coord`, `cfg.eval.surface_range`. Esses valores precisam bater entre o server e o que o checkpoint `m2t2.pth` foi treinado — não inventar/alterar sem necessidade.
  - Confirmado: `num_points=16384`, `mask_thresh=0.4`, `object_thresh=0.4`, `world_coord=True`, `surface_range=0.02`.
- [x] Confirmar `gripper_depth` default usado em `build_6d_grasp` (`m2t2/action_decoder.py`, hoje hardcoded `0.1034`). Vai ser referência pro ajuste do gripper do Spot na Parte 2. Adotar o valor hardcoded por hora.
  - Confirmado hardcoded em `build_6d_grasp` (m2t2/action_decoder.py:31), mantido sem alteração.

### 1.2 Criar `client-server/m2t2_server.py`
- [x] Criar pasta `client-server/` na raiz do repo M2T2 (mesma convenção do GraspGen).
- [x] Escrever `build_data_from_arrays(xyz, rgb, cfg)`: substitui `load_rgb_xyz` (que só lê de disco: `meta_data.pkl`, `rgb.png`, `depth.npy`, `seg.png`). Deve receber `xyz` (Nx3, já em world coordinates) e `rgb` (Nx3, já normalizado como `normalize_rgb` faria) em memória, e montar o dict `data` com `inputs`, `points`, `task='pick'`, `object_inputs` (placeholder, não usado em pick), `ee_pose=eye(4)`, `cam_pose`.
  - Também inclui `bottom_center=zeros(3)`: necessário porque o branch de placement do M2T2 roda sempre (`num_place_queries=8` no config, independente da task), e `model.infer` lê `data['bottom_center']` incondicionalmente.
- [x] Escrever `load_model(checkpoint_path, model_cfg)`: `M2T2.from_config(cfg.m2t2)` + `load_state_dict` + `.cuda().eval()`. Chamado **uma única vez** no startup do processo, não por request.
- [x] Escrever `infer(model, xyz, rgb, cfg)`: replica o loop de `cfg.eval.num_runs` do `load_and_predict` original (`demo.py`), usando `sample_points` + `collate` + `to_gpu`/`to_cpu` + `model.infer(batch, cfg.eval)`, mas **sem** nenhuma chamada a `create_visualizer`/`meshcat` — esse é o ponto que travava o `demo.py` esperando o `meshcat-server` subir. O server headless não deve ter essa dependência em nenhum caminho de execução.
- [x] Implementar o loop principal `zmq.REP` com as 3 actions:
  - `{"action": "health"}` → `{"status": "ok"}`
  - `{"action": "metadata"}` → `{"gripper_name": "franka_panda_2f", "model_name": "m2t2"}`
  - `{"action": "infer", "points": <Nx3>, "rgb": <Nx3>, "num_runs": int, "mask_thresh": float}` → `{"grasps": [...], "confidence": [...], "contacts": [...]}`
- [x] Serialização: `msgpack` + `msgpack_numpy.patch()` para os arrays NumPy (mesmo padrão do `graspgen_server`).
- [x] Adicionar `pyzmq`, `msgpack`, `msgpack-numpy` ao `requirements.txt` (ou instalar direto no Dockerfile) — não fazem parte das deps originais do M2T2.

### 1.3 Ajustar o Dockerfile
- [x] Trocar o `CMD` atual (ou o que `run.sh` chama) de `demo.py ...` para:
  `python3 client-server/m2t2_server.py --checkpoint /checkpoints/m2t2.pth --port 5556`
- [x] Confirmar que o `CMD` novo **não** depende de `meshcat-server` estar rodando — remover/ajustar qualquer linha do `run.sh` que sobe o meshcat, já que o server headless não usa.
- [x] Rebuild da imagem (`docker/build.sh`) e smoke test manual do container isolado (rodar, checar log de "modelo carregado" sem erro, sem travar).
  - Gotcha adicional encontrado e corrigido: `m2t2/` não tem `__init__.py` (namespace package), então só era importável quando o script rodava a partir da raiz do repo (caso do `demo.py`). Rodar `client-server/m2t2_server.py` quebrava com `ModuleNotFoundError: No module named 'm2t2'` porque o diretório do script (`client-server/`) vira `sys.path[0]`, não a raiz. Corrigido com `ENV PYTHONPATH=/workspace/m2t2` no Dockerfile.

### 1.4 Smoke test do server isolado (antes de plugar no ROS)
- [x] Escrever um client Python solto de teste (`client-server/m2t2_client_test.py`), análogo ao `graspgen_client.py` de referência: carrega uma nuvem de pontos de `sample_data/real_world/00` (via os mesmos helpers `load_rgb_xyz`, só pra montar o teste), monta o request `infer`, manda via ZMQ, imprime quantos grasps voltaram.
- [x] Rodar esse client contra o `m2t2_server` subido standalone (sem compose ainda) e confirmar que o número de grasps retornado é compatível com o que `demo.py` já validou (`object_00 has 7 grasps`, mesmo checkpoint/config).
  - Confirmado como compatível, com ressalva: contagem de objetos/grasps varia bastante entre execuções (testado com 3 chamadas diretas ao `load_and_predict` original do `demo.py`, sem passar pelo server: de 2 a 4 objetos detectados, grasps por objeto entre 1 e 83). Isso é estocasticidade normal do modelo — cada run resample aleatoriamente 16384 de ~500k pontos brutos via `sample_points` — não um bug do server. O client de teste (via server) produziu resultados no mesmo formato e mesma ordem de grandeza.

---

## Parte 2 — Pacote ROS 2 (`grasp_baseline/`)

path: 
```bash
nexus@nexus-PC:~/Desktop/isaac_dl_grasp (baseline_host)$ ls baseline/ros-ws/src/grasp_baseline/grasp_baseline/
debug_perception.py      __init__.py  sam3_node.py                 utils
graspgen_bridge_node.py  __pycache__  sim_graspgen_bridge_node.py
```
### 2.1 Novo bridge node
- [ ] Copiar `graspgen_bridge_node.py` → `m2t2_bridge_node.py` como ponto de partida.
- [ ] Trocar o endpoint ZMQ (porta `5556`, já que GraspGen e M2T2 não sobem simultaneamente — ver decisão anterior de comentar o `graspgen_server` no compose).
- [ ] Ajustar `parse_grasp_response`: adaptar pro schema de saída do M2T2 (`grasps`: lista de matrizes 4x4 por objeto; `confidence`; `contacts`) — schema diferente do GraspGen, não reaproveitar o parser 1:1.
- [ ] Decidir e implementar a estratégia de seleção de objeto: M2T2 no modo pick **não precisa de nuvem pré-segmentada** — ele processa a cena inteira e propõe objetos internamente via `objectness`. Duas opções, escolher uma:
  - (a) Continuar usando `sam3_node` para filtrar, **depois** da inferência, qual proposta do M2T2 corresponde ao objeto-alvo (overlap espacial entre máscara SAM3 e `grasp_contacts` de cada proposta).
  - (b) Mandar a cena inteira sem filtro prévio e usar todas as propostas do M2T2 diretamente (fluxo mais "end-to-end", mas perde a seleção por prompt de objeto).
- [ ] Publicar mensagens, integrar com cuRobo IK/MotionGen e collision filter — reaproveitar a lógica existente do `graspgen_bridge_node` sem alterações (compartilhada entre os dois bridges).

### 2.2 Config de gripper
- [ ] Criar `config/m2t2_gripper.yaml`, análogo ao `spot_gripper.yaml` do GraspGen — **não copiar os valores**, recalcular. Base de partida: `gripper_depth=0.1034` (constante do Franka Panda hardcoded em `build_6d_grasp`) precisa ser substituída pela distância física equivalente do gripper do Spot.
- [ ] Validar a convenção de eixos da pose 4x4 retornada pelo M2T2 (`contact_dir`, `approach_dir`, ver `build_6d_grasp`) contra a convenção que o `spot_gripper.yaml` já assume — confirmar se é a mesma base (conversa anterior levantou hipótese de que sim, mas não foi verificado linha a linha ainda).

### 2.3 Launch e infra
- [ ] Criar `launch/m2t2_bridge.launch.py`, espelhando `graspgen_bridge.launch.py`.
- [ ] Adicionar recipe `just ros-m2t2` em `.just/baseline.just` (ou equivalente).
- [ ] `docker-compose.yaml`: adicionar serviço `m2t2_server` (imagem `m2t2:latest`, volume dos checkpoints, `shm_size`, reserva de GPU); comentar/remover o serviço `graspgen_server`; atualizar `depends_on` do `isaac_ros_zed` para `m2t2_server`.

### 2.4 Teste end-to-end
- [ ] Subir `docker compose up` com `m2t2_server` + `isaac_ros_zed`.
- [ ] Rodar `sam3_node` + `m2t2_bridge_node` juntos numa cena real, confirmar poses chegando em `/joint_states`/`arm_joint_command` sem erro de parsing.
- [ ] Comparar qualitativamente com uma execução equivalente do GraspGen (mesma cena, se possível) como primeira checagem de sanidade do baseline.

---

## Notas / riscos conhecidos (não bloqueantes, mas documentar no estudo)
- M2T2 foi treinado/avaliado só em cenários tabletop; generalização pra outros contextos do Spot é zero-shot e não garantida.
- Constantes geométricas da garra (offset, `gripper_depth`) são específicas do Franka Panda — precisão da adaptação pro Spot impacta diretamente a qualidade dos grasps, não só a integração de software.