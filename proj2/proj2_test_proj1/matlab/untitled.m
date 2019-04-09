u  = load('test_99_state.txt');
ux = u(:,2)./u(:,1);
[idx, uxmax] = sort(ux)

u(4760,:)
u(2923,:)
u(uxmax(1240),:)
u(uxmax(3617),:)
u(uxmax(3800),:)

u(uxmax(6266),:)
u(1905,:)
u(5993,:)
