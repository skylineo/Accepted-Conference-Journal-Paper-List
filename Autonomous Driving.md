# Autonomous Driving Applicaiton Catagory

## Perception
### Object Detection
#### 2D Object Detection
#### 3D Object Detection
##### lidar based
##### lidar&image fusion based
| title | abstract | year |
|---------|---------|---------|
| Motion and geometry-related information fusion through a framework for object identification from a moving camera in urban driving scenarios | 行1单元 | partc 2023 |
| 行2单元 | 行2单元 | 行2单元 |
| 行3单元 | 行3单元 | 行3单元 |

### Object Tracking
#### Single Object Tracking
#### Multiple Object tracking
| 列1标题 | 列2标题 | 列3标题 |
|---------|---------|---------|
| Cooperative multi-camera vehicle tracking and traffic surveillance with edge artificial intelligence and representation learning | 行1单元 | partc 2023 |
| Using spatiotemporal stacks for precise vehicle tracking from roadside 3D LiDAR data | 行2单元 | partc 2023 |
| 行3单元 | 行3单元 | 行3单元 |

### Localization
#### Sensor-Based Localization
##### Camera Based
##### Lidar Based
##### Radar Based
##### GPS/IMU Based
#### Cooperative Localization
##### V2V Based
##### V2I Based

### Segmentation
#### Semantic Segmentation
#### Instance Segmentation

### Motion forecasting
#### vehicle prediction
| 列1标题 | 列2标题 | 列3标题 |
|---------|---------|---------|
|TrajGAIL: Generating urban vehicle trajectories using generative adversarial imitation learning|.This study proposes TrajGAIL, a generative adversarial imitation learning (GAIL) model for urban vehicle trajectory data.This study proposes a new approach that combines a partially-observable Markov decision process (POMDP) within the GAIL framework. POMDP can map the sequence of location observations into a latent state, thereby allowing more generalization of the state definition and incorporating the information of previously visited locations in modeling the vehicle’s next locations. In summary, the generation procedure of urban vehicle trajectories in TrajGAIL is formulated as an imitation learning problem based on POMDP, which can effectively deal with sequential data, and this imitation learning problem is solved using GAIL, which enables trajectory generation that can scale to large road network environments. (将轨迹预测任务转化为序列决策任务，基于POMDP，通过GAIL，将generator作为轨迹生成器，将Discriminator作为奖励函数，采用模仿学习进行训练)|
|Joint prediction of next location and travel time from urban vehicle trajectories using long short-term memory neural networks|This paper aims to incorporate travel time prediction in the next location prediction problem to enable the prediction of the city-wide movement trajectory of an individual vehicle by considering both where the vehicle will go next and when it will arrive. We propose two deep learning models based on long short-term memory (LSTM) neural networks with self-attention mechanism—namely, hybrid LSTM and sequential LSTM. These models capture patterns in location and time sequences in trajectory data and their dependencies to predict next locations and travel times simultaneously.(直接采用LSTM网络进行next goal预测，在输入的信息中添加了时间信息)|partc 2021|
|Injecting knowledge in data-driven vehicle trajectory predictors |In this work, we propose a knowledge-driven(model based) and data-driven methods for trajectory prediction. The knowledge-driven (KD) trajectory is achieved by any knowledge-driven model. The KD output and the information from scenarios are then taken as inputs by our Realistic Residual Block (RRB) and the residuals is the output and required to be added to KD trajectory are found. In other words, the KD trajectory estimates coarse-grained behaviors based on the common driving performances while the residuals address fine-grained behaviors coming from non-modeled social interactions in KD trajectory as well as the long-tail of performances. This structure allows imposing knowledge by any function while allowing the residual block to be trainable. Moreover, in order not to diminish the feasibility of KD prediction, we finally constrain the final output with Model Predictive Control (MPC) to ensure kinematic-feasibility of predictions（基于物理模型生成KD Trajectory，将KDT和各类环境信息输入到神经网络中拟合出每个点的μ和θ，即偏移量的概率分布，从而得到多模态轨迹分布）|partc 2021| 
|A deep learning framework for modelling left-turning vehicle behaviour considering diagonal-crossing motorcycle conflicts at mixed-flow intersections|To model vehicle turning behaviour, we present a novel interaction-aware deep-learning framework. Firstly, a Long Short-Term Memory (LSTM) based network is employed to encode vehicle historical motion features. Then, each vehicle’s potential target lanes are identified with a probabilistic method, followed by a pooling module that extracts and summarizes intention features. Thirdly, Graph Attention Network (GAT) and a synthesized network are introduced to model vehicle-vehicle interaction and vehicle-motorcycle interaction respectively. Finally, multiple kinds of obtained features are sent to a LSTM based decoder module to generate future displacement and body orientation.（将道路看作意图，goal based？）|partc 2021|
|A Bayesian inference based adaptive lane change prediction model|This paper aims to build a lane change prediction model for surrounding vehiclesl to adapt to the change of road and traffic environment based on machine learning methods. The lane change prediction model contains a basic model and an adaptive model: the basic model is a long short-term memory (LSTM) based prediction model which reflects the decision-making mode for drivers; the adaptive prediction model embeds an adaptive decision threshold on the basic model, and the threshold updates by Bayesian Inference method on time.（基于LSTM进行未来行为的二元预测，考虑到环境的动态特性，对实时环境进行观测量，采用贝叶斯过程对分布大阈值进行跟新）|partc 2021|
|Predicting time-varying, speed-varying dilemma zones using machine learning and continuous vehicle tracking| This paper proposes an innovative framework of predicting driver behavior under varying dilemma zone conditions.A linear SVM was used to extract through vehicles from all approaching vehicles detected from radar sensors. A hierarchical clustering method was utilized to classify different traffic patterns by time-of-day. Finally, driver behavior prediction models were developed using three machine learning techniques (i.e., linear SVM, polynomial SVM, and ANN) widely adopted for binary classification problems to predicts drivers’ stop-or-go decision.（基于机器学习的行为分类）|partc 2021|
|A discrete-continuous multi-vehicle anticipation model of driving behaviour in heterogeneous disordered traffic conditions |
| Towards explainable motion prediction using heterogeneous graph representations | 行1单元 | partc 2023 |
| Real-time forecasting of driver-vehicle dynamics on 3D roads: A deep-learning framework leveraging Bayesian optimisation | 行2单元 | partc 2023 |
| Dynamic-learning spatial-temporal Transformer network for vehicular trajectory prediction at urban intersections | 行3单元 | partc 2023 |
| A two-layer integrated model for cyclist trajectory prediction considering multiple interactions with the environment | 行1单元 | partc 2023 |
| A physics-informed Transformer model for vehicle trajectory prediction on highways | 行2单元 | partc 2023 |
| Explainable multimodal trajectory prediction using attention models | 行3单元 | partc 2022 |
|  Robust unsupervised learning of temporal dynamic vehicle-to-vehicle interactions | 行1单元 | partc 2022 |
| Are socially-aware trajectory prediction models really socially-aware? | 行2单元 | partc 2022 |
| Fine-grained highway autonomous vehicle lane-changing trajectory prediction based on a heuristic attention-aided encoder-decoder model | 行3单元 | partc 2022 |
| Long-term 4D trajectory prediction using generative adversarial networks | 行1单元 | partc 2022 |
| An improved learning-based LSTM approach for lane change intention prediction subject to imbalanced data | To deal the problem of the imbalance between the lane changing data and lane keeping data, we propose a hierarchical over-sampling bagging method to generate more diverse and informative instances of minority classes for training LSTM model. Furthermore, we also propose a sampling technique to keep the temporal information and consider the interaction among agents through concatenating their trajectories when constructing features.（本文的主要创新点在数据生成上） | partc 2021 |
| 行3单元 | 行3单元 | 行3单元 |
| 行1单元 | 行1单元 | 行1单元 |
| 行2单元 | 行2单元 | 行2单元 |
| 行3单元 | 行3单元 | 行3单元 |   
#### pedestrian prediction
| 列1标题 | 列2标题 | 列3标题 |
|---------|---------|---------|
| A context-aware pedestrian trajectory prediction framework for automated vehicles | 行1单元 | partc 2022 |
|Pedestrian intention prediction: A convolutional bottom-up multi-task approach |We want to predict the pedestrians’ intention to cross the road as early as possible given a single image. We present a new neural network for the primary task of pedestrian intention prediction. Our model takes as input only a single RGB image and generates a map predicting the probability that each pixel constitutes a pedestrian who is either crossing or not, bypassing the need for a people detector and running at constant time. We additionally have our model output in parallel, the detailed human body pose for each pedestrian to show that our network can be easily extended to perform a variety of other tasks with little overhead. The byproduct of the intention or pose map also allows the model to function as a generic people detector.(采样神经网络实现基于像素点的行为预测及body pose)|partc 2021|
|Decoding pedestrian and automated vehicle interactions using immersive virtual reality and interpretable deep learning|an interpretable machine learning framework is proposed to explore factors affecting pedestrians’ wait time before crossing mid-block crosswalks in the presence of automated vehicles.. Pedestrian wait time behaviour is then analysed using a data-driven Cox Proportional Hazards (CPH) model, in which the linear combination of the covariates is replaced by a flexible non-linear deep neural network.(本文没有进行轨迹预测，而是通过CPH模型对影响行人过斑马线的因素进行了定量分析，并采用类似于梯度传播的SHAP值分析进行可解释性分析）|partc 2021|
| 行2单元 | 行2单元 | 行2单元 |
| 行3单元 | 行3单元 | 行3单元 | 

### Risk Assessment(Risk field) 
| 列1标题 | 列2标题 | 列3标题 |
|---------|---------|---------|
|  A risk field-based metric correlates with driver’s perceived risk in manual and automated driving: A test-track study | 行1单元 | partc 2021 |
|Risk assessment based collision avoidance decision-making for autonomous vehicles in multi-scenarios|(1)A probabilistic approach of risk assessment that considered both driving safety and driving style for CA was proposed and verified to be effective in multi-scenarios. (2) Multiple safety indicators were comprehensively used to guarantee safety from multiple aspects, which addressed the demerits of the individual indicators in previous studies. (3) The collision avoidance strategy with adjustable driving style preferences to meet the demand of different consumers was developed to improve drivers’ acceptance of CA systems.|partc 2021|
| 行2单元 | 行2单元 | 行2单元 |
| 行3单元 | 行3单元 | 行3单元 |



## Decision Making & Planning
### Decision Making
| 列1标题 | 列2标题 | 列3标题 |
|---------|---------|---------|
|Modeling aggressive driving behavior based on graph construction |partc 2022|  
|Modified DDPG car-following model with a real-world human driving experience with CARLA simulator| (partc 2023)|  
|Why did the AI make that decision? Towards an explainable artificial intelligence (XAI) for autonomous driving systems|(partc 2023)| 
|Human-like decision making for lane change based on the cognitive map and hierarchical reinforcement learning|(partc 2023)|  
|Improve generalization of driving policy at signalized intersections with adversarial learning|(partc 2023)|  
|Deep Reinforcement Learning for Personalized Driving Recommendations to Mitigate Aggressiveness and Riskiness: Modeling and Impact Assessment|(partc 2022)|  
|Decision making of autonomous vehicles in lane change scenarios: Deep reinforcement learning approaches with risk awareness|(partc 2022)|  
|Hierarchical and game-theoretic decision-making for connected and automated vehicles in overtaking scenarios|(partc 2023)|  
|Toward personalized decision making for autonomous vehicles: A constrained multi-objective reinforcement learning technique|(partc 2023)|  

### Motion Planning
| 列1标题 | 列2标题 | 列3标题 |
|---------|---------|---------|
|Connected automated vehicle control in single lane roundabouts|To optimize the trajectory of connected automated vehicles (CAVs) in roundabouts, we formulate an trajectory optimization problem that includes vehicle dynamics and collision-avoidance constraints. The objective function of the problem minimizes the distance of CAVs to their destinations and their acceleration magnitudes. The methodology also involves a customized solution technique that convexifies the collision-avoidance constraints and decompose the convexified problem into two sub-problems. The first sub-problem only includes vehicle dynamics constraints while the second sub-problem projects the solutions of the first sub-problem onto a collision-free region. These sub-problem is then transformed into a quadratic problem to get the optimal solution by redefining its decision variables along vehicle paths.(构建目标函数：最小化和目标间距离以及加速度变化大小，考虑碰撞约束。采样泰勒展开将问题凸化并将凸问题分解为连分子问题，问题1仅考虑车辆动力学约束，问题2在问题1结果上考虑为无碰撞约束，两个问题可采用类似物二次规划的方法求解) |partc 2021|
|Reliable trajectory-adaptive routing strategies in stochastic, time-varying networks with generalized correlations| 
|Predictive trajectory planning for autonomous vehicles at intersections using reinforcement learning|(partc 2023)|  
|A deep inverse reinforcement learning approach to route choice modeling with context-dependent rewards|(partc 2023)|  
|No more road bullying: An integrated behavioral and motion planner with proactive right-of-way acquisition capability|(partc 2023)|  
|Autonomous navigation at unsignalized intersections: A coupled reinforcement learning and model predictive control approach|(partc 2022)|  
|A Markov Decision Process framework to incorporate network-level data in motion planning for connected and automated vehicles|(partc 2022)|  
|Optimization-based path-planning for connected and non-connected automated vehicles|(partc 2022)|  
|Optimizing operations at freeway weaves with connected and automated vehicles|1. Development of a new methodology for optimizing the CAV trajectories in a system with full CAV market penetration to maximize the weaving segment capacity and minimize the delay for each vehicle. 2. The optimization model formulates a novel logic which, unlike the majority of the literature, is independent of the characteristics of the leading vehicle. 3. The optimization algorithm enables the early departure of vehicles, which are not necessarily those that entered the system first. In other words, the FIFO queue order assumptions are relaxed.(基于V2I信息，实现车辆在高速公路编制路段的分段的path planning and optimizing,重点在于优化模型中的目标函数和约束条件的设计)|partc 2021|




## Control
### lateral control
| 列1标题 | 列2标题 | 列3标题 |
|---------|---------|---------|
|CLACD: A complete LAne-Changing decision modeling framework for the connected and traditional environments| This study develops a DLC model (discretionary lane-changing) for the traditional environment, which is extended for the connected environment. An integrated approach is employed to model DLC behavior by combining the target lane selection using the utility theory approach and the gap acceptance behavior using a game theory approach. (采用效用理论来选择自主换道的目标道路，之后对于换到另一道的间隙接受行为分析时，采用两玩家非零和非合作博弈模型来实现建模，SV将等待下一个可用间隙所需的加速度作为奖励函数目标，FV1的奖励则定义为为了阻止或促进DLC所需的加速度。)|partc 2021|
|Space-weighted information fusion using deep reinforcement learning: The context of tactical control of lane-changing autonomous vehicles and connectivity range assessment|To migirate the limit caused by their sensor range, we describe a Deep reinforcement Learning based approach that integrates the data collected through sensing and connectivity capabilities from other vehicles located in the proximity of the CAV and from those located further downstream, and we use the fused data to guide lane changing. In addition, recognizing the importance of the connectivity range (CR) to the performance of not only the algorithm but also of the vehicle in the actual driving environment.(采用DRL，在输入状态信息时不仅包括传感器感受到的牵前车的信息，还包括四周连接范围内的V2V的信息)|partc 2021|
|Predicting and explaining lane-changing behaviour using machine learning: A comparative study(partc 2022)|  
### longitudinal control
| 列1标题 | 列2标题 | 列3标题 |
|---------|---------|---------|
|A discrete-continuous multi-vehicle anticipation model of driving behaviour in heterogeneous disordered traffic conditions |This study proposes a multi-vehicle anticipation-based discrete-continuous choice modelling framework for car following in heterogeneous disordered traffic (HDT) conditions. To incorporate multi-vehicle anticipation, the concept of an influence zone around a vehicle is introduced. Vehicles within the influence zone can potentially influence the subject vehicle’s driving behaviour. Further, driving decisions are characterized as combination of discrete and continuous components. The discrete component involves the decision to accelerate, decelerate, or maintain constant speed and the continuous component involves the decision of how much to accelerate or decelerate. A copula-based joint modelling framework that allows dependencies between discrete and continuous components is proposed.(基于V2V来获取感知范围内车辆的状态信息，基于状态信息进行lane change or not的分类，基于分类结果采用控制器输出相应的控制参数的大小) |partc 2021|
|On multi-class automated vehicles: Car-following behavior and its implications for traffic dynamics|This paper develops a unifying framework to unveil the physical car-following (CF) behaviors of automated vehicles (AVs) under different control paradigms and parameter settings. The proposed framework adopts the flexible asymmetric behavior (AB) model to reveal the control mechanisms and their manifestation in the physical CF behavior, particularly their response to traffic disturbances. A mapping relationship between the AB model parameters and control parameters is then obtained to understand the range of CF behavior possible. Finally, a predictive modeling approach based on a logistic classifier coupled with a convoluted Multivariate Gaussian Process (MGP) is designed to predict the CF behavior of an AV to prevent risk.(采用AB模型来构建车辆的跟车模型，并在其中加入参数θ来获得对环境自适应的时间间隔和空间间隔，将AB模型转化为控制器，采用预测模型来预测未来跟车操作实现对风险的预估)|partc 2021|
|A physics-informed deep learning paradigm for car-following models |This paper aims to develop a family of neural network based car-following models that are informed by physics-based models, which leverage the advantage of both physics-based and deep learning based models. We design physics-informed deep learning car following model (PIDL-CF) architectures encoded with 4 popular physics-based models. Acceleration is predicted for 4 traffic regimes: acceleration, deceleration, cruising, and emergency braking while considering the physics constraint.(即在神经网络中考虑物理约束，用约束函数、约束网络来实现) |partc 2021|
|Model predictive control policy design, solutions, and stability analysis for longitudinal vehicle control considering shockwave damping(partc 2023)|  
|Unified framework for over-damped string stable adaptive cruise control systems(partc 2023)|  
|Congestion-mitigating MPC design for adaptive cruise control based on Newell’s car following model: History outperforms prediction(partc 2022)|  
|Multianticipation for string stable Adaptive Cruise Control and increased motorway capacity without vehicle-to-vehicle communication(partc 2022)|  
|Physics-augmented models to simulate commercial adaptive cruise control (ACC) systems(partc 2022)|  
|Comfortable and energy-efficient speed control of autonomous vehicles on rough pavements using deep reinforcement learning(partc 2022)|  
|A generative car-following model conditioned on driving styles(partc 2022)|  
|Significance of low-level control to string stability under adaptive cruise control: Algorithms, theory and experiments(partc 2022)|  
|Safety-critical traffic control by connected automated vehicles(partc 2023)|  
### simultaneous lateral and longitudinal control
| 列1标题 | 列2标题 | 列3标题 |
|---------|---------|---------|
|Learning how to dynamically route autonomous vehicles on shared roads|We study a dynamic routing game, in which the route choices of autonomous cars can be controlled and the human drivers react selfishly and dynamically. As the problem is prohibitively large, we use deep reinforcement learning to learn a policy for controlling the autonomous vehicles. This policy indirectly influences human drivers to route themselves in such a way that minimizes congestion on the network.|partc 2021|
|Markov-game modeling of cyclist-pedestrian interactions in shared spaces: A multi-agent adversarial inverse reinforcement learning approach|This study proposes a novel Multi-Agent Adversarial Inverse Reinforcement Learning approach (MA-AIRL) to model and simulate road user interactions at shared space facilities. Unlike the traditional game-theoretic framework that models multi-agent systems as a single time-step payoff, the proposed approach is based on Markov Games (MG) which models road users’ sequential decisions concurrently.The proposed algorithm recovers road users’ multi-agent reward functions using adversarial deep neural network discriminators and estimates their optimal policies using Multi-agent Actor-Critic with Kronecker factors (MACK) deep reinforcement learning.(采用GAN网络进行奖励函数的生成，奖励函数中考虑到多种车之间交互。此外构建MARL，实现多车控制序列的同时生成) |partc 2021|
|Trajectory planning for connected and automated vehicles at isolated signalized intersections under mixed traffic environment| This study proposes an approach to the decentralized planning of CAV trajectories at an isolated signalized intersection under the mixed traffic environment. A bi-level optimization model is formulated based on discrete time to optimize both the longitudinal and lateral trajectories of a single CAV given signal timings and the trajectory information of surrounding vehicles. The upper-level model optimizes lateral lane-changing strategies. The lower-level model optimizes longitudinal acceleration profiles based on the lane-changing strategies from the upper-level model. A Lane-Changing Strategy Tree (LCST) and a Parallel Monte-Carlo Tree Search (PMCTS) algorithm are designed to solve the bi-level optimization model.(用到V2V信息，采用分层控制，上层实施是否换道的决策，下层基于上层决策进行纵向加速度动作生成)|partc 2021|
|Automated eco-driving in urban scenarios using deep reinforcement learning |The innovation of this paper lies in the use of Deep Reinforcement Learning (RL) to develop eco-driving strategies for autonomous vehicles in urban environments. This approach considers the challenges of implementing eco-driving strategies with insufficient information about the vehicles ahead and achieves effective eco-driving under these constraints.(基于强化学习中TD3算法来实现车辆的eco-control)|partc 2021|

 

 
 
 



##  Cooperation Driving
### Centralized Coordination
_definityion:One all-knowing, leading vehicle is responsible for the planning, coordination, and synchronization of the maneuvers of all the vehicles._
| 列1标题 | 列2标题 | 列3标题 |
|---------|---------|---------|
|A novel hierarchical cooperative merging control model of connected and automated vehicles featuring flexible merging positions in system optimization(partc 2022)|
|Cooperative path and trajectory planning for autonomous vehicles on roads without lanes: A laboratory experimental demonstration(partc 2022)|
|Connected automated vehicle cooperative control with a deep reinforcement learning approach in a mixed traffic environment(partc 2021)|To solve the car following problem in mixed traffic, we decomposed mixed traffic into multiple subsystems where each subsystem is comprised of a human-driven vehicle (HDV) followed by cooperative CAVs. Based on that, a cooperative CAV control strategy is developed based on a DRL algorithm, enabling CAVs to learn the leading HDV’s characteristics and make longitudinal control decisions cooperatively to improve the performance of each subsystem locally and consequently enhance performance for the whole mixed traffic flow.(将整个交通路段划分为one HV+multi-AV,使得AV的状态取学习HV的状态实现连号的跟车行为。采用DPPO，多车状态分散收集更新后对全局策略进行跟新。基于MPC（预定义期望状态+物理约束）进行奖励函数的设计。|partc 2021|
|Centralized vehicle trajectory planning on general platoon sorting problem with multi-vehicle lane changing(partc 2023)|
|Coordinated trajectory planning for lane-changing in the weaving areas of dedicated lanes for connected and automated vehicles(partc 2021)|
|Coordinated lane-changing scheduling of multilane CAV platoons in heterogeneous scenarios(partc 2023)|
|Reinforcement Learning based cooperative longitudinal control for reducing traffic oscillations and improving platoon stability|
|Fair collaborative vehicle routing: A deep multi-agent reinforcement learning approach(partc 2023)|
### Decentralized with Coordination
_definition:This allows the vehicles to directly communicate with all neighboring vehicles having access to local knowledge aiming to plan maneuvers._
| 列1标题 | 列2标题 | 列3标题 |
|---------|---------|---------|
|Constraint-tree-driven modeling and distributed robust control for multi-vehicle cooperation at unsignalized intersections|
|A control strategy for merging a single vehicle into a platoon at highway on-ramps(partc 2022)|
|Safe autonomous lane changes and impact on traffic flow in a connected vehicle environment(partc 2023)|
|A platoon-based cooperative optimal control for connected autonomous vehicles at highway on-ramps under heavy traffic(partc 2023)|
|Make space to change lane: A cooperative adaptive cruise control lane change controller(partc 2022)|
|A general approach to smoothing nonlinear mixed traffic via control of autonomous vehicles(partc 2023)|
|Lane change scheduling for connected and autonomous vehicles(partc 2023) |   
|Distributed model predictive control for heterogeneous vehicle platoon with unknown input of leading vehicle(partc 2023)|  
|A platoon-based cooperative optimal control for connected autonomous vehicles at highway on-ramps under heavy traffic(partc 2023)|  
|A deep reinforcement learning based distributed control strategy for connected automated vehicles in mixed traffic platoon(partc 2023)|  
|Distributed cooperative trajectory and lane changing optimization of connected automated vehicles: Freeway segments with lane  drop(partc 2022)|  
|Decentralized motion planning for intelligent bus platoon based on hierarchical optimization framework(partc 2023)|  
|Connected and automated vehicle distributed control for on-ramp merging scenario: A virtual rotation approach|It propose a virtual rotation approach to simplify the  merging problem to a virtual car following (CF) problem. The rotation process, which serves as an upper-level controller, uses a predetermine merge point as a reference to calculate the relative spacing for vehicle and then determines the virtual car following sequences of vehicles in a predefined merging control area. A lower-level cooperative distributed control strategy is proposed to control vehicles’ trajectories with a specifically designed unidirectional multi-leader communication topology.（本文将merging场景转化为car following，基于跟车任务进行多车离散控制）| partc 2021|
|Flow-aware platoon formation of Connected Automated Vehicles in a mixed traffic with human-driven vehicles|  
|Cooperative signal-free intersection control using virtual platooning and traffic flow regulation(partc 2022)|  
|Modeling decentralized mandatory lane change for connected and autonomous vehicles: An analytical method | The framework achieves a safe MLC(Mandatory lane change) maneuver with multiple-CAV cooperation, including the lane change times, positions, and trajectories for the CAVs, while minimizing the impacts of the MLC maneuver on overall traffic. This problem was rigorously formulated and solved by an analytical method that significantly decreases computational time and renders the methodology suitable for practical implementations.（不了解）|partc 2021| 
|Robust optimal control of connected and automated vehicle platoons through improved particle swarm optimization(partc 2022)|
|Distributed data-driven predictive control for cooperatively smoothing mixed traffic flow(partc 2023)|
|Fault-Tolerant cooperative driving at highway on-ramps considering communication failure(partc 2023)|
### Decentralized without Coordination
_defifnition:The vehicles can observe other vehicles in the neighborhood without having the potential to exchange the information._
| 列1标题 | 列2标题 | 列3标题 |
|---------|---------|---------|
### Centralized and Decentralized mix-Based
| 列1标题 | 列2标题 | 列3标题 |
|---------|---------|---------|
|COOR-PLT: A hierarchical control model for coordinating adaptive platoons of connected and autonomous vehicles at signal-free intersections based on deep reinforcement learning|(partc 2023)|  
|Analysis of cooperative driving strategies at road network level with macroscopic fundamental diagram(partc 2022)|  
|Formation control with lane preference for connected and automated vehicles in multi-lane scenarios(partc 2022)|  
|A cooperative driving framework for urban arterials in mixed traffic conditions|The proposed framework combines centralized and distributed control concepts, where the infrastructure generates optimal signal timing plans and provides high-level trajectory guidance to the CAVs while detailed trajectories are generated by each vehicle. The system consists of three levels of models. At the vehicle level, a state transition diagram is designed for different modes of operations of CAVs including eco-trajectory planning, cooperative adaptive cruise control (CACC) and collision avoidance. At the intersection level, a mixed-integer linear programming (MILP) problem is formulated to optimize the signal timing plan and arrival time of CAVs, with 
consideration of CACC platooning behaviors. At the corridor level, link performance functions are applied to calculate the total delay of the coordinated phases of each intersection, and a linear programming (LP) problem is formulated to optimize the offsets for every cycle, which are then passed to the intersection level.(基于车辆所处的状态，构建状态转换图实现面对不同的场景时采用不同的decentralized策略（目标函数+约束条件，如CACC，CA），基于交通流量、车辆等待时间、绿灯时间等因素对信号灯进行centralized控制，基于信号灯对车辆进行CAV行为的优化使得等待红灯时间最短，之后基于LP对交通信号周期偏移进行求解，保证整个交通走廊的流通性）|partc 2021|

  
 

## Vehicle-level(End2End)
## Vehicle-Road(V2X)
### V2I
| 列1标题 | 列2标题 | 列3标题 |
|---------|---------|---------|
|An eco-driving algorithm based on vehicle to infrastructure (V2I) communications for signalized intersections|(partc 2022)|
|Data-driven road side unit location optimization for connected-autonomous-vehicle-based intersection control| To achieve low vehicle-to-road-side-unit (V2R) communication delay and support the implementation of CAV-based intersection control strategies, this study addresses the problem of road side unit (RSU) location optimization at a single intersection. The problem is formulated as a two-stage stochastic mixed-integer nonlinear program. The model aims to minimize the sum of the cost associated with RSU investment and the expectation of the penalty cost associated with V2R communication delay exceeding a pre-determined threshold. The first stage of the program determines the number and location of RSUs, when the intersection control strategy to be implemented is unknown. Given the first stage decision and the implemented intersection control strategy, the second stage model optimizes the detection area allocation among RSUs to minimize the penalty cost.|partc 2021|
|On the deployment of V2X roadside units for traffic prediction |we focus on traffic predictions that instead are based on recording the trajectories of CVs and processing them via the connected roadside infrastructure, and investigate how V2I communication may facilitate traffic prediction. In particular, we establish metrics to quantify the amount of traffic prediction. We utilize analytical and numerical tools to evaluate these metrics as a function of (i) the location of the roadside units along the road, (ii) the communication range of the roadside units, and (iii) the penetration rate of connected vehicles on the road.|partc 2021|
### V2V&V2P
| 列1标题 | 列2标题 | 列3标题 |
|---------|---------|---------|
|A local traffic characteristic based dynamic gains tuning algorithm for cooperative adaptive cruise control considering wireless communication delay|(partc 2022)|



## Vehicle-driver
| 列1标题 | 列2标题 | 列3标题 |
|---------|---------|---------|
|Multi-scale driver behavior modeling based on deep spatial-temporal representation for intelligent vehicles|To enhance mutual understanding between driver and vehicle, the driver behavior recognition system is designed to simultaneously recognize the driver’s physical and mental behaviours with different time scales.The encoder module is designed based on a deep convolutional neural network (CNN) to capture spatial information from the input video stream. Then, several decoders for different driver states estimation are proposed with fully-connected (FC) and long short-term memory (LSTM) based recurrent neural networks (RNN).|partc 2021| 
|Toward human-vehicle collaboration: Review and perspectives on human-centered collaborative automated driving| In this study, literature review and perspectives on the human behaviors and cognition (HBC) for ADVs toward human-autonomy (H-A) collaboration are proposed. First, the H-A collaboration basics and key factors are reviewed. Then, the HBC issues in driver behavior modeling and understanding are discussed. Specifically, two key factors are reviewed, which are human trust and situation awareness (SA). Next, HBC in two H-A collaboration-enabled vehicle control methods, namely, shared control and take-over control, are analyzed. |partc 2021|



## Others
### Generative

### Traffic flow operation
 Stability analysis and connected vehicles management for mixed traffic flow with platoons of connected automated vehicles(partc 2023)
 MSGNN: A Multi-structured Graph Neural Network model for real-time incident prediction in large traffic networks(partc 2023)
 Adaptive control with moving actuators at motorway bottlenecks with connected and automated vehicles(partc 2023)
 Nonlinear model predictive control of large-scale urban road networks via average speed control(partc 2023)
 Optimal internal boundary control of lane-free automated vehicle traffic
 Traffic scheduling and control in fully connected and automated networks
 A safety-enhanced eco-driving strategy for connected and autonomous vehicles: A hierarchical and distributed framework(partc 2023)
 A novel spatio-temporal generative inference network for predicting the long-term highway traffic speed(partc 2023)
 Decentralized signal control for multi-modal traffic network: A deep reinforcement learning approach(partc 2023)
 Alpha-fair large-scale urban network control: A perimeter control based on a macroscopic fundamental diagram(partc 2023)
 TD3LVSL: A lane-level variable speed limit approach based on twin delayed deep deterministic policy gradient in a connected automated vehicle environment(partc 2023)
 A scenario-based distributed model predictive control approach for freeway networks(partc 2022)
 Connected automated vehicle trajectory optimization along signalized arterial: A decentralized approach under mixed traffic environment(partc 2022)
 Lane change detection and prediction using real-world connected vehicle data(partc 2022)
 Does LSTM outperform 4DDTW-KNN in lane change identification based on eye gaze data?(partc 2022)
 Short-term traffic prediction using physics-aware neural networks(partc 2022)
 CVLight: Decentralized learning for adaptive traffic signal control with connected vehicles(partc 2022)
 Multi-agent reinforcement learning for Markov routing games: A new modeling paradigm for dynamic traffic assignment(partc 2022)
 DDP-GCN: Multi-graph convolutional network for spatiotemporal traffic forecasting(partc 2022)
 Nonlinear model predictive control of large-scale urban road networks via average speed control(partc 2023)
 Adaptive control with moving actuators at motorway bottlenecks with connected and automated vehicles(partc 2023)
 DRL-TP3: A learning and control framework for signalized intersections with mixed connected automated traffic

