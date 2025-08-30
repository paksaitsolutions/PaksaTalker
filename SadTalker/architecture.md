graph LR
    A[Audio File] --> B(Audio Preprocessing Module)
    B --> C{Audio Features}
    D[Image] --> E(SadTalker Module)
    C --> E
    E --> F{Head Pose, Facial Expressions}
    C --> G(Gesture Generation Module)
    G --> H{Body Pose Sequence}
    F --> I(Combining Module)
    H --> I
    I --> J[Full-Body Video]