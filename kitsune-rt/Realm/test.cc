#include<realm.h>
#include<stdint.h>

extern "C" {
Realm::Barrier createRealmBarrier(); 
void realmInitRuntime(int argc, char**argv); 
void realmSpawn(Realm::Processor::TaskFuncPtr func, 
		  const void* args, 
		  size_t arglen
      ); 
void realmCall(Realm::Processor::TaskFuncPtr func, 
		  const void* args, 
		  size_t arglen
      ); 
void realmSync(Realm::Barrier b); 
size_t realmGetNumProcs(); 
  typedef struct context {
    Realm::Runtime rt;
    std::set<Realm::Event> events;
    std::vector<Realm::Processor> procs;
    Realm::Processor procgroup;
    size_t numprocs;
    unsigned cur_task;
  } context;
context getRealmCTX(); 
}

void task(const void* args, uint64_t arglen, const void* mem, uint64_t memlen, Realm::Processor p){
  int i = *(int*)args;
  printf("hello from task %d / %d\n", i, realmGetNumProcs()); 
}

void taskcaller(const void* args, uint64_t arglen, const void* mem, uint64_t memlen, Realm::Processor p){
  auto b = createRealmBarrier();
  auto n = realmGetNumProcs(); 
  int a[n]; for(int i=0; i < n; i++) a[i] = i;  
  for(int i=0; i<n; i++){
    realmSpawn(&task, (void*)(a+i), sizeof(int));
  }
  realmSync(b); 
}

int main(int argc, char** argv){
  realmInitRuntime(argc, argv); 
  realmCall(&taskcaller, nullptr, 0); 
}
